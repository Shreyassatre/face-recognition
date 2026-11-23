from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .settings import get_settings


settings = get_settings()
is_sqlite = settings.database_url.startswith("sqlite")
if is_sqlite:
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if is_sqlite else {}, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_local_schema() -> None:
    """Idempotent schema migrations for both PostgreSQL (Neon) and local SQLite."""
    if not is_sqlite:
        with engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE employees ADD COLUMN IF NOT EXISTS location VARCHAR(200)")
            connection.exec_driver_sql("ALTER TABLE employees DROP CONSTRAINT IF EXISTS employees_tenant_id_external_id_key")
            connection.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_employees_tenant_org_external "
                "ON employees (tenant_id, organization_id, external_id)"
            )
        return
    with engine.begin() as connection:
        def add_column(table: str, column: str, definition: str) -> None:
            existing = connection.exec_driver_sql(f"PRAGMA table_info({table})").mappings().all()
            if not any(item["name"] == column for item in existing):
                connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

        add_column("employees", "organization_id", "INTEGER")
        add_column("employees", "location", "VARCHAR(200)")
        add_column("verification_sessions", "organization_id", "INTEGER")
        add_column("attendance_events", "organization_id", "INTEGER")
        tenant_columns = connection.exec_driver_sql("PRAGMA table_info(tenants)").mappings().all()
        if not any(column["name"] == "organization_id" for column in tenant_columns):
            connection.exec_driver_sql("ALTER TABLE tenants ADD COLUMN organization_id VARCHAR(128)")
            connection.exec_driver_sql("UPDATE tenants SET organization_id = 'org-' || id WHERE organization_id IS NULL")
            connection.exec_driver_sql("CREATE UNIQUE INDEX ix_tenants_organization_id ON tenants (organization_id)")
        columns = connection.exec_driver_sql("PRAGMA table_info(verification_sessions)").mappings().all()
        employee_column = next((column for column in columns if column["name"] == "employee_id"), None)
        if not employee_column or not employee_column["notnull"]:
            return
        connection.exec_driver_sql("ALTER TABLE verification_sessions RENAME TO verification_sessions_legacy")
        connection.exec_driver_sql("""
            CREATE TABLE verification_sessions (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                tenant_id INTEGER NOT NULL REFERENCES tenants(id),
                organization_id INTEGER REFERENCES organizations(id),
                employee_id INTEGER REFERENCES employees(id),
                action VARCHAR(20) NOT NULL,
                return_origin VARCHAR(500),
                expires_at DATETIME NOT NULL,
                used_at DATETIME,
                created_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("""
            INSERT INTO verification_sessions
              (id, tenant_id, organization_id, employee_id, action, return_origin, expires_at, used_at, created_at)
            SELECT id, tenant_id, organization_id, employee_id, action, return_origin, expires_at, used_at, created_at
            FROM verification_sessions_legacy
        """)
        connection.exec_driver_sql("DROP TABLE verification_sessions_legacy")
        connection.exec_driver_sql("CREATE INDEX ix_verification_sessions_tenant_id ON verification_sessions (tenant_id)")
        connection.exec_driver_sql("CREATE INDEX ix_verification_sessions_employee_id ON verification_sessions (employee_id)")
        connection.exec_driver_sql("CREATE INDEX ix_verification_sessions_expires_at ON verification_sessions (expires_at)")
