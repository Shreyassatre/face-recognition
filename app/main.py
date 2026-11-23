from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
import uuid
import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import Base, engine, get_db, migrate_local_schema
from .face_engine import FaceEngine, FaceValidationError, cosine_similarity
from .models import AttendanceEvent, Employee, Organization, Tenant, VerificationSession
from .settings import get_settings

settings = get_settings()
app = FastAPI(title="Hosted Face Attendance", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=[settings.hosted_origin], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
_face_engine: FaceEngine | None = None
STATIC_DIR = Path(__file__).parent / "static"


def utcnow() -> datetime:
    return datetime.utcnow()


def get_engine() -> FaceEngine:
    global _face_engine
    if _face_engine is None:
        _face_engine = FaceEngine(settings.model_dir)
    return _face_engine


def bootstrap() -> None:
    Base.metadata.create_all(engine)
    migrate_local_schema()
    with Session(engine) as db:
        if not db.query(Tenant).filter_by(client_key=settings.default_client_key).first():
            db.add(Tenant(organization_id="local-demo-org", name="Local demo organization", client_key=settings.default_client_key))
            db.commit()
        client = db.query(Tenant).filter_by(client_key=settings.default_client_key).first()
        if not db.query(Organization).filter_by(client_id=client.id, organization_id="local-demo-org").first():
            db.add(Organization(client_id=client.id, organization_id="local-demo-org", name="Local demo organization"))
            db.commit()


@app.on_event("startup")
def startup() -> None:
    bootstrap()


def unauthorized(detail: str = "Invalid client credentials") -> HTTPException:
    return HTTPException(status_code=401, detail=detail)


def client_tenant(x_client_key: str = Header(...), db: Session = Depends(get_db)) -> Tenant:
    tenant = db.query(Tenant).filter_by(client_key=x_client_key).first()
    if not tenant:
        raise unauthorized()
    return tenant


def require_admin(x_admin_key: str = Header(...)) -> None:
    if x_admin_key != settings.admin_key:
        raise unauthorized("Invalid admin credentials")


class EmployeeCreate(BaseModel):
    organization_id: str = Field(min_length=1, max_length=128)
    organization_name: str = Field(min_length=1, max_length=200)
    user_id: str = Field(min_length=1, max_length=128)
    user_name: str = Field(min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=200)


class EmployeeUpdate(BaseModel):
    user_name: str | None = Field(default=None, min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=200)
    organization_id: str | None = Field(default=None, min_length=1, max_length=128)
    organization_name: str | None = Field(default=None, min_length=1, max_length=200)


class ReEnrollRequest(BaseModel):
    return_origin: str | None = None


class ClientApplicationCreate(BaseModel):
    application_name: str = Field(min_length=2, max_length=100)


class SessionCreate(BaseModel):
    action: str = Field(pattern="^(check_in|check_out)$")
    organization_id: str | None = Field(default=None, min_length=1, max_length=128)
    return_origin: str | None = None


class EnrollmentSessionCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    organization_id: str | None = Field(default=None, min_length=1, max_length=128)
    return_origin: str | None = None


class VerificationConfirm(BaseModel):
    signature: str = Field(min_length=10, max_length=4096)


class OrganizationSelection(BaseModel):
    selection_token: str = Field(min_length=10, max_length=8192)
    organization_id: str = Field(min_length=1, max_length=128)


def validate_return_origin(return_origin: str | None) -> str | None:
    if not return_origin:
        return None
    normalized = return_origin.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise HTTPException(status_code=422, detail="return_origin must be a valid http(s) origin")
    if settings.app_env.lower() == "production" and parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise HTTPException(status_code=422, detail="Production return_origin must use https")
    allowlist = settings.allowed_return_origin_set
    if allowlist and normalized not in allowlist:
        raise HTTPException(status_code=403, detail="return_origin is not in ALLOWED_RETURN_ORIGINS")
    return normalized


class DevSessionCreate(BaseModel):
    organization_id: str = Field(min_length=1, max_length=128)
    organization_name: str = Field(min_length=1, max_length=100)
    employee_id: str | None = Field(default=None, max_length=128)
    display_name: str | None = Field(default=None, max_length=200)
    action: str = Field(pattern="^(enroll|check_in|check_out)$")


class SessionPublic(BaseModel):
    action: str
    expires_at: str
    return_origin: str | None = None


def get_organization(db: Session, client: Tenant, organization_id: str, organization_name: str | None = None) -> Organization:
    organization = db.query(Organization).filter_by(client_id=client.id, organization_id=organization_id).first()
    if not organization and organization_name:
        organization = Organization(client_id=client.id, organization_id=organization_id, name=organization_name)
        db.add(organization); db.commit(); db.refresh(organization)
    if not organization:
        raise HTTPException(status_code=404, detail="Organization was not found for this application key")
    return organization


def encode_session(session: VerificationSession) -> str:
    return jwt.encode({"sid": session.id, "exp": session.expires_at.replace(tzinfo=timezone.utc), "iat": datetime.now(timezone.utc)}, settings.app_secret, algorithm="HS256")


def new_session(db: Session, tenant: Tenant, organization: Organization | None, action: str, employee: Employee | None = None, return_origin: str | None = None) -> dict:
    if action == "enroll" and (not employee or not organization):
        raise HTTPException(status_code=422, detail="Enrollment sessions require an employee and organization")
    verification = VerificationSession(
        id=str(uuid.uuid4()), tenant_id=tenant.id, organization_id=organization.id if organization else None, employee_id=employee.id if employee else None, action=action,
        return_origin=return_origin.rstrip("/") if return_origin else None,
        expires_at=utcnow() + timedelta(seconds=settings.session_ttl_seconds),
    )
    db.add(verification)
    db.commit()
    token = encode_session(verification)
    return {"token": token, "expires_at": verification.expires_at.isoformat() + "Z", "url": f"{settings.hosted_origin}/verify/{token}"}


def session_from_token(token: str, db: Session) -> VerificationSession:
    try:
        payload = jwt.decode(token, settings.app_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="This verification link is invalid or has expired.")
    verification = db.get(VerificationSession, payload.get("sid"))
    if not verification or verification.expires_at < utcnow():
        raise HTTPException(status_code=401, detail="This verification link has expired.")
    if verification.used_at:
        raise HTTPException(status_code=409, detail="This verification link has already been used.")
    return verification


def authorization_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing verification token")
    return authorization[7:]


def reject_verification(code: str, message: str, guidance: str, status_code: int = 422) -> None:
    """Return user-safe, actionable feedback without exposing model thresholds or internals."""
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message, "guidance": guidance})


def explain_face_error(error: FaceValidationError) -> None:
    reason = str(error)
    if reason.startswith("No face found"):
        reject_verification("face_not_visible", "We could not see your face.", "Center your face in the guide and try again.")
    if reason.startswith("Exactly one face"):
        reject_verification("multiple_faces", "More than one face is visible.", "Make sure only you are in the camera frame.")
    if reason.startswith("Camera resolution"):
        reject_verification("camera_resolution", "Your camera image is too small.", "Use a higher-resolution camera or move closer.")
    if reason.startswith("Move closer"):
        reject_verification("face_too_small", "Your face is too far from the camera.", "Move closer until your face fits inside the guide.")
    if reason.startswith("The image is blurry"):
        reject_verification("frame_blurry", "We could not get a clear image.", "Hold still for a moment, then try again.")
    reject_verification("capture_invalid", "We could not use this camera image.", "Check your camera and try again.")


@app.get("/health")
def health():
    try:
        get_engine()
        model_status = "ready"
    except RuntimeError as error:
        model_status = str(error)
    return {"status": "ok", "face_models": model_status}


@app.get("/v1/public/config")
def public_config():
    """Non-sensitive UI capabilities only; never expose environment secrets."""
    return {"development_launcher_enabled": settings.app_env.lower() == "development"}


@app.post("/v1/platform/api-clients", dependencies=[Depends(require_admin)])
def create_client_application(payload: ClientApplicationCreate, db: Session = Depends(get_db)):
    """Platform-only endpoint: issue one API key to an integrating application, e.g. Buyscrapapp."""
    application_id = payload.application_name.strip().lower().replace(" ", "-")
    if db.query(Tenant).filter_by(organization_id=application_id).first():
        raise HTTPException(status_code=409, detail="An API key already exists for this application")
    key = f"fa_{uuid.uuid4().hex}{uuid.uuid4().hex[:8]}"
    client = Tenant(organization_id=application_id, name=payload.application_name.strip(), client_key=key)
    db.add(client); db.commit()
    return {"application_id": application_id, "application_name": client.name, "api_key": key, "warning": "Store this key in the Buyscrapapp server environment; it is returned only by this call."}


def serialize_employee(employee: Employee, organization: Organization | None) -> dict:
    return {
        "user_id": employee.external_id,
        "user_name": employee.display_name,
        "location": employee.location,
        "organization_id": organization.organization_id if organization else None,
        "organization_name": organization.name if organization else None,
        "enrolled": employee.embedding is not None,
        "enrolled_at": (employee.enrolled_at.isoformat() + "Z") if employee.enrolled_at else None,
        "created_at": (employee.created_at.isoformat() + "Z") if employee.created_at else None,
    }


def load_scoped_employee(db: Session, tenant: Tenant, user_id: str, scoped_org_id: str | None, target_org_id: str | None = None) -> tuple[Employee, Organization | None]:
    effective_org = scoped_org_id or target_org_id
    query = db.query(Employee, Organization).outerjoin(Organization, Employee.organization_id == Organization.id).filter(
        Employee.tenant_id == tenant.id,
        Employee.external_id == user_id,
    )
    if effective_org:
        query = query.filter(Organization.organization_id == effective_org)
    rows = query.all()
    if not rows:
        if scoped_org_id:
            exists_elsewhere = db.query(Employee).filter_by(tenant_id=tenant.id, external_id=user_id).first()
            if exists_elsewhere:
                raise HTTPException(status_code=403, detail="You do not have permission to manage users in this organization")
        raise HTTPException(status_code=404, detail="Employee was not found")
    if len(rows) > 1:
        raise HTTPException(status_code=409, detail="Multiple organizations match this user_id; specify organization_id")
    return rows[0]


@app.get("/v1/admin/organizations")
def list_organizations(tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    organizations = db.query(Organization).filter_by(client_id=tenant.id).order_by(Organization.name.asc()).all()
    employees = db.query(Employee).filter_by(tenant_id=tenant.id).all()
    counts: dict[int, list[int]] = {}
    for emp in employees:
        if emp.organization_id is None:
            continue
        pair = counts.setdefault(emp.organization_id, [0, 0])
        pair[0] += 1
        if emp.embedding is not None:
            pair[1] += 1
    return [
        {
            "organization_id": org.organization_id,
            "organization_name": org.name,
            "employee_count": counts.get(org.id, [0, 0])[0],
            "enrolled_count": counts.get(org.id, [0, 0])[1],
        }
        for org in organizations
    ]


@app.get("/v1/admin/employees")
def list_employees(
    organization_id: str | None = None,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    effective_org_id = x_organization_id.strip() if x_organization_id and x_organization_id.strip() else (organization_id.strip() if organization_id and organization_id.strip() else None)
    query = db.query(Employee, Organization).outerjoin(Organization, Employee.organization_id == Organization.id).filter(Employee.tenant_id == tenant.id)
    if effective_org_id:
        query = query.filter(Organization.organization_id == effective_org_id)
    rows = query.order_by(Employee.created_at.desc()).all()
    return [serialize_employee(emp, org) for emp, org in rows]


@app.post("/v1/admin/employees")
def create_employee(
    payload: EmployeeCreate,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    if x_organization_id and x_organization_id.strip() and payload.organization_id.strip() != x_organization_id.strip():
        raise HTTPException(status_code=403, detail="You can only enroll users in your own organization")
    organization = get_organization(db, tenant, payload.organization_id.strip(), payload.organization_name.strip())
    external_id = payload.user_id.strip()
    clean_location = payload.location.strip() if payload.location and payload.location.strip() else None
    existing = db.query(Employee).filter_by(tenant_id=tenant.id, organization_id=organization.id, external_id=external_id).first()
    if existing:
        if existing.embedding is not None:
            raise HTTPException(status_code=409, detail="This user_id is already enrolled in this organization")
        existing.display_name = payload.user_name.strip()
        existing.location = clean_location
        db.commit()
        db.refresh(existing)
        return serialize_employee(existing, organization)
    employee = Employee(
        tenant_id=tenant.id,
        organization_id=organization.id,
        external_id=external_id,
        display_name=payload.user_name.strip(),
        location=clean_location,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return serialize_employee(employee, organization)


@app.patch("/v1/admin/employees/{user_id}")
def update_employee(
    user_id: str,
    payload: EmployeeUpdate,
    organization_id: str | None = None,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    scoped_org = x_organization_id.strip() if x_organization_id and x_organization_id.strip() else None
    target_org = organization_id.strip() if organization_id and organization_id.strip() else None
    employee, organization = load_scoped_employee(db, tenant, user_id, scoped_org, target_org)
    if payload.user_name is not None:
        employee.display_name = payload.user_name.strip()
    if "location" in payload.model_fields_set:
        employee.location = payload.location.strip() if payload.location and payload.location.strip() else None
    if payload.organization_id is not None:
        new_org_id = payload.organization_id.strip()
        if scoped_org and new_org_id != scoped_org:
            raise HTTPException(status_code=403, detail="Organization admins cannot move users to another organization")
        organization = get_organization(db, tenant, new_org_id, payload.organization_name or new_org_id)
        employee.organization_id = organization.id
    db.commit()
    db.refresh(employee)
    return serialize_employee(employee, organization)


@app.post("/v1/admin/employees/{user_id}/re-enroll")
def re_enroll_employee(
    user_id: str,
    payload: ReEnrollRequest,
    organization_id: str | None = None,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    scoped_org = x_organization_id.strip() if x_organization_id and x_organization_id.strip() else None
    target_org = organization_id.strip() if organization_id and organization_id.strip() else None
    employee, organization = load_scoped_employee(db, tenant, user_id, scoped_org, target_org)
    if not organization:
        raise HTTPException(status_code=409, detail="Employee is not assigned to an organization")
    return_origin = validate_return_origin(payload.return_origin)
    employee.embedding = None
    employee.enrolled_at = None
    db.commit()
    return new_session(db, tenant, organization, "enroll", employee, return_origin)


@app.delete("/v1/admin/employees/{user_id}")
def delete_employee(
    user_id: str,
    organization_id: str | None = None,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    scoped_org = x_organization_id.strip() if x_organization_id and x_organization_id.strip() else None
    target_org = organization_id.strip() if organization_id and organization_id.strip() else None
    employee, _ = load_scoped_employee(db, tenant, user_id, scoped_org, target_org)
    db.query(VerificationSession).filter_by(employee_id=employee.id).delete()
    db.query(AttendanceEvent).filter_by(employee_id=employee.id).delete()
    db.delete(employee)
    db.commit()
    return {"deleted": True, "user_id": user_id}


@app.post("/v1/sessions")
def create_session(payload: SessionCreate, tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    return_origin = validate_return_origin(payload.return_origin)
    # When organization_id is provided, scope strictly to that org; otherwise match across all orgs for this client key.
    organization = get_organization(db, tenant, payload.organization_id) if payload.organization_id else None
    return new_session(db, tenant, organization, payload.action, return_origin=return_origin)


@app.post("/v1/admin/enrollment-sessions")
def create_enrollment_session(
    payload: EnrollmentSessionCreate,
    x_organization_id: str | None = Header(default=None),
    tenant: Tenant = Depends(client_tenant),
    db: Session = Depends(get_db),
):
    scoped_org = x_organization_id.strip() if x_organization_id and x_organization_id.strip() else None
    target_org = payload.organization_id.strip() if payload.organization_id and payload.organization_id.strip() else None
    employee, organization = load_scoped_employee(db, tenant, payload.user_id.strip(), scoped_org, target_org)
    if employee.embedding:
        raise HTTPException(status_code=409, detail="Employee is already enrolled")
    if not organization:
        raise HTTPException(status_code=409, detail="Employee is not assigned to an organization")
    return_origin = validate_return_origin(payload.return_origin)
    return new_session(db, tenant, organization, "enroll", employee, return_origin)


@app.post("/v1/verifications/confirm")
def confirm_verification(payload: VerificationConfirm, tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    """Server-to-server confirmation: customer backend verifies a popup result JWT before saving attendance."""
    try:
        claims = jwt.decode(payload.signature, settings.app_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or tampered verification signature")
    organization = db.query(Organization).filter_by(client_id=tenant.id, organization_id=claims.get("organization_id")).first()
    if not organization:
        raise HTTPException(status_code=403, detail="Verification does not belong to this application key")
    action = claims.get("action")
    if action in {"check_in", "check_out"}:
        event = db.query(AttendanceEvent).filter_by(
            tenant_id=tenant.id,
            organization_id=organization.id,
            verification_id=claims.get("verification_id"),
            action=action,
        ).first()
        if not event:
            raise HTTPException(status_code=404, detail="Attendance verification record was not found")
    elif action == "enroll":
        employee = db.query(Employee).filter_by(
            tenant_id=tenant.id,
            organization_id=organization.id,
            external_id=claims.get("user_id"),
        ).first()
        if not employee or not employee.embedding:
            raise HTTPException(status_code=404, detail="Enrollment record was not found")
    else:
        raise HTTPException(status_code=400, detail="Unsupported verification action")
    return {"verified": True, "result": claims}


@app.post("/v1/dev/sessions")
def create_development_session(payload: DevSessionCreate, db: Session = Depends(get_db)):
    """Convenience endpoint for local testing only. Deliberately unavailable outside development."""
    if settings.app_env.lower() != "development":
        raise HTTPException(status_code=404, detail="Not found")
    tenant = db.query(Tenant).filter_by(client_key=settings.default_client_key).first()
    organization = get_organization(db, tenant, payload.organization_id, payload.organization_name)
    if payload.action != "enroll":
        return new_session(db, tenant, organization, payload.action, return_origin=settings.hosted_origin)
    if not payload.employee_id or not payload.display_name:
        raise HTTPException(status_code=422, detail="Employee ID and name are required only for enrollment.")
    employee = db.query(Employee).filter_by(organization_id=organization.id, external_id=payload.employee_id).first()
    if not employee:
        employee = Employee(tenant_id=tenant.id, organization_id=organization.id, external_id=payload.employee_id, display_name=payload.display_name)
        db.add(employee)
        db.commit()
        db.refresh(employee)
    if employee.embedding:
        raise HTTPException(status_code=409, detail="This employee is already enrolled. Create a check-in/out session instead.")
    return new_session(db, tenant, organization, "enroll", employee, settings.hosted_origin)


@app.get("/v1/session")
def public_session(token: str = Depends(authorization_token), db: Session = Depends(get_db)) -> SessionPublic:
    verification = session_from_token(token, db)
    return SessionPublic(action=verification.action, expires_at=verification.expires_at.isoformat() + "Z", return_origin=verification.return_origin)


def build_signed_result(organization: Organization, employee: Employee, action: str, outcome: str, verification_id: str, similarity: float) -> dict:
    result = {
        "status": outcome,
        "organization_id": organization.organization_id,
        "organization_name": organization.name,
        "user_id": employee.external_id,
        "display_name": employee.display_name,
        "employee_id": employee.external_id,
        "location": employee.location,
        "action": action,
        "occurred_at": utcnow().isoformat() + "Z",
        "verification_id": verification_id,
        "similarity": round(float(similarity), 4),
    }
    result["signature"] = jwt.encode(result, settings.app_secret, algorithm="HS256")
    return result


@app.post("/v1/verify")
async def verify(request: Request, image: UploadFile = File(...), token: str = Depends(authorization_token), db: Session = Depends(get_db)):
    verification = session_from_token(token, db)
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Please submit a JPEG, PNG, or WebP camera frame.")
    raw = await image.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Camera frame is too large.")
    try:
        extracted = get_engine().extract(raw)
    except FaceValidationError as error:
        explain_face_error(error)
    except RuntimeError:
        reject_verification("service_unavailable", "Verification is temporarily unavailable.", "Please try again in a moment.", status_code=503)
    verification_id = str(uuid.uuid4())
    similarity = 1.0
    if verification.action == "enroll":
        employee = db.get(Employee, verification.employee_id)
        if not employee:
            raise HTTPException(status_code=409, detail="The enrollment target is unavailable.")
        if employee.embedding:
            raise HTTPException(status_code=409, detail="Employee is already enrolled")
        employee.embedding = extracted.embedding.tobytes()
        employee.enrolled_at = utcnow()
        outcome = "enrolled"
        organization = db.get(Organization, verification.organization_id)
    elif verification.organization_id is not None:
        candidates = db.query(Employee).filter(Employee.organization_id == verification.organization_id, Employee.embedding.is_not(None)).all()
        if not candidates:
            reject_verification("no_enrolled_users", "No enrolled users are available for this organization.", "Ask your administrator to enroll your face first.", status_code=409)
        scored = sorted(
            ((cosine_similarity(extracted.embedding, np.frombuffer(candidate.embedding, dtype=np.float32)), candidate) for candidate in candidates),
            key=lambda item: item[0], reverse=True,
        )
        similarity, employee = scored[0]
        if similarity < settings.match_threshold:
            reject_verification("identity_not_confirmed", "We could not confirm your identity.", "Face the camera directly and try again.")
        if len(scored) > 1 and similarity - scored[1][0] < settings.match_margin:
            reject_verification("match_ambiguous", "We could not confidently identify one person.", "Move closer, keep your face centered, and try again.")
        organization = db.get(Organization, verification.organization_id)
        event = AttendanceEvent(id=str(uuid.uuid4()), tenant_id=verification.tenant_id, organization_id=organization.id, employee_id=employee.id,
                                action=verification.action, similarity=similarity, verification_id=verification_id,
                                user_agent=request.headers.get("user-agent"))
        db.add(event)
        outcome = "verified"
    else:
        candidates = db.query(Employee).filter(
            Employee.tenant_id == verification.tenant_id,
            Employee.organization_id.is_not(None),
            Employee.embedding.is_not(None),
        ).all()
        if not candidates:
            reject_verification("no_enrolled_users", "No enrolled users are available.", "Ask your administrator to enroll your face first.", status_code=409)
        by_org: dict[int, list[Employee]] = {}
        for candidate in candidates:
            by_org.setdefault(candidate.organization_id, []).append(candidate)
        matched_records: list[tuple[float, Organization, Employee]] = []
        had_ambiguous = False
        for org_pk, org_candidates in by_org.items():
            scored = sorted(
                ((cosine_similarity(extracted.embedding, np.frombuffer(candidate.embedding, dtype=np.float32)), candidate) for candidate in org_candidates),
                key=lambda item: item[0], reverse=True,
            )
            best_sim, best_emp = scored[0]
            if best_sim < settings.match_threshold:
                continue
            if len(scored) > 1 and best_sim - scored[1][0] < settings.match_margin:
                had_ambiguous = True
                continue
            org = db.get(Organization, org_pk)
            if org:
                matched_records.append((best_sim, org, best_emp))
        matched_records.sort(key=lambda item: item[0], reverse=True)
        if matched_records:
            top_sim = matched_records[0][0]
            matched_records = [item for item in matched_records if top_sim - item[0] <= 0.12]
        if not matched_records:
            if had_ambiguous:
                reject_verification("match_ambiguous", "We could not confidently identify one person.", "Move closer, keep your face centered, and try again.")
            reject_verification("identity_not_confirmed", "We could not confirm your identity.", "Face the camera directly and try again.")
        if len(matched_records) > 1:
            selection_token = jwt.encode(
                {
                    "sid": verification.id,
                    "purpose": "select_org",
                    "exp": verification.expires_at.replace(tzinfo=timezone.utc),
                    "options": [
                        {"org_pk": org.id, "organization_id": org.organization_id, "emp_pk": emp.id, "similarity": round(float(sim), 4)}
                        for sim, org, emp in matched_records
                    ],
                },
                settings.app_secret,
                algorithm="HS256",
            )
            return {
                "status": "select_organization",
                "action": verification.action,
                "selection_token": selection_token,
                "matches": [
                    {
                        "organization_id": org.organization_id,
                        "organization_name": org.name,
                        "user_id": emp.external_id,
                        "display_name": emp.display_name,
                        "similarity": round(float(sim), 4),
                    }
                    for sim, org, emp in matched_records
                ],
            }
        similarity, organization, employee = matched_records[0]
        verification.organization_id = organization.id
        event = AttendanceEvent(id=str(uuid.uuid4()), tenant_id=verification.tenant_id, organization_id=organization.id, employee_id=employee.id,
                                action=verification.action, similarity=similarity, verification_id=verification_id,
                                user_agent=request.headers.get("user-agent"))
        db.add(event)
        outcome = "verified"
    verification.used_at = utcnow()
    db.commit()
    return build_signed_result(organization, employee, verification.action, outcome, verification_id, similarity)


@app.post("/v1/verify/select-organization")
def select_organization(request: Request, payload: OrganizationSelection, token: str = Depends(authorization_token), db: Session = Depends(get_db)):
    verification = session_from_token(token, db)
    if verification.action not in {"check_in", "check_out"}:
        raise HTTPException(status_code=400, detail="Organization selection is only valid for check-in or check-out.")
    try:
        claims = jwt.decode(payload.selection_token, settings.app_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Selection token is invalid or has expired.")
    if claims.get("purpose") != "select_org" or claims.get("sid") != verification.id:
        raise HTTPException(status_code=401, detail="Selection token does not match this verification session.")
    chosen = next((opt for opt in claims.get("options", []) if opt.get("organization_id") == payload.organization_id), None)
    if not chosen:
        raise HTTPException(status_code=403, detail="Selected organization was not matched for this face.")
    organization = db.get(Organization, chosen["org_pk"])
    employee = db.get(Employee, chosen["emp_pk"])
    if not organization or not employee:
        raise HTTPException(status_code=404, detail="Selected organization or employee is unavailable.")
    verification_id = str(uuid.uuid4())
    similarity = float(chosen["similarity"])
    event = AttendanceEvent(
        id=str(uuid.uuid4()),
        tenant_id=verification.tenant_id,
        organization_id=organization.id,
        employee_id=employee.id,
        action=verification.action,
        similarity=similarity,
        verification_id=verification_id,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(event)
    verification.organization_id = organization.id
    verification.used_at = utcnow()
    db.commit()
    return build_signed_result(organization, employee, verification.action, "verified", verification_id, similarity)


@app.get("/v1/admin/attendance", dependencies=[Depends(require_admin)])
def attendance(tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    rows = db.query(AttendanceEvent, Employee).join(Employee, AttendanceEvent.employee_id == Employee.id).filter(AttendanceEvent.tenant_id == tenant.id).order_by(AttendanceEvent.occurred_at.desc()).limit(100).all()
    return [{"event_id": event.id, "employee_id": employee.external_id, "employee_name": employee.display_name, "action": event.action, "similarity": event.similarity, "occurred_at": event.occurred_at.isoformat() + "Z"} for event, employee in rows]


@app.get("/{frontend_path:path}", include_in_schema=False)
def hosted_frontend(frontend_path: str):
    """Serve the compiled hosted UI and fall back to index.html for SPA verification routes."""
    if not STATIC_DIR.is_dir():
        raise HTTPException(status_code=503, detail="Hosted frontend has not been built. Run npm.cmd run build in frontend/.")
    candidate = (STATIC_DIR / frontend_path).resolve()
    static_root = STATIC_DIR.resolve()
    if candidate.is_relative_to(static_root) and candidate.is_file():
        return FileResponse(candidate)
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="Hosted frontend build is incomplete.")
    return FileResponse(index)
