from datetime import datetime, timedelta, timezone
from pathlib import Path
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
    user_name: str = Field(min_length=1, max_length=200)


class ClientApplicationCreate(BaseModel):
    application_name: str = Field(min_length=2, max_length=100)


class SessionCreate(BaseModel):
    action: str = Field(pattern="^(check_in|check_out)$")
    organization_id: str = Field(min_length=1, max_length=128)
    return_origin: str | None = None


class EnrollmentSessionCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    return_origin: str | None = None


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


def new_session(db: Session, tenant: Tenant, organization: Organization, action: str, employee: Employee | None = None, return_origin: str | None = None) -> dict:
    if action == "enroll" and not employee:
        raise HTTPException(status_code=422, detail="Enrollment sessions require an employee")
    verification = VerificationSession(
        id=str(uuid.uuid4()), tenant_id=tenant.id, organization_id=organization.id, employee_id=employee.id if employee else None, action=action,
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


@app.post("/v1/admin/employees")
def create_employee(payload: EmployeeCreate, tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    organization = get_organization(db, tenant, payload.organization_id, payload.organization_name)
    employee = Employee(tenant_id=tenant.id, organization_id=organization.id, external_id=str(uuid.uuid4()), display_name=payload.user_name)
    db.add(employee)
    db.commit()
    return {"user_id": employee.external_id, "user_name": employee.display_name, "organization_id": organization.organization_id, "enrolled": False}


@app.post("/v1/sessions")
def create_session(payload: SessionCreate, tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    if payload.return_origin and not payload.return_origin.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="return_origin must be an http(s) origin")
    # The client key selects the organization. No person identifier enters an attendance token.
    organization = get_organization(db, tenant, payload.organization_id)
    return new_session(db, tenant, organization, payload.action, return_origin=payload.return_origin)


@app.post("/v1/admin/enrollment-sessions")
def create_enrollment_session(payload: EnrollmentSessionCreate, tenant: Tenant = Depends(client_tenant), db: Session = Depends(get_db)):
    employee = db.query(Employee).filter_by(tenant_id=tenant.id, external_id=payload.user_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee was not found")
    if employee.embedding:
        raise HTTPException(status_code=409, detail="Employee is already enrolled")
    if payload.return_origin and not payload.return_origin.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="return_origin must be an http(s) origin")
    organization = db.get(Organization, employee.organization_id)
    return new_session(db, tenant, organization, "enroll", employee, payload.return_origin)


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
    else:
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
        event = AttendanceEvent(id=str(uuid.uuid4()), tenant_id=verification.tenant_id, organization_id=verification.organization_id, employee_id=employee.id,
                                action=verification.action, similarity=similarity, verification_id=verification_id,
                                user_agent=request.headers.get("user-agent"))
        db.add(event)
        outcome = "verified"
    verification.used_at = utcnow()
    db.commit()
    organization = db.get(Organization, verification.organization_id)
    result = {"status": outcome, "organization_id": organization.organization_id, "organization_name": organization.name, "user_id": employee.external_id,
              "display_name": employee.display_name, "employee_id": employee.external_id, "action": verification.action,
              "occurred_at": utcnow().isoformat() + "Z", "verification_id": verification_id,
              "similarity": round(similarity, 4)}
    result["signature"] = jwt.encode(result, settings.app_secret, algorithm="HS256")
    return result


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
