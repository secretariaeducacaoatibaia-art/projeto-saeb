import hashlib
import jwt
import uvicorn
import secrets
import json
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import os

from fastapi import (
    FastAPI, Depends, HTTPException, status, Request, Form, Cookie, Query, Response
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, desc
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.sql.expression import func as sql_func

# --- Configurações Globais ---
SECRET_KEY = "chave-secreta-saeb-2025-muito-segura"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480
PAGE_SIZE = 20

# --- Configuração do Banco de Dados ---
SQLALCHEMY_DATABASE_URL = "sqlite:////var/data/saeb.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Modelos do Banco de Dados (SQLAlchemy) ---
class Superuser(Base):
    __tablename__ = "superusers"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    password_hash = Column(String(255))

class School(Base):
    __tablename__ = "schools"
    id = Column(Integer, primary_key=True, index=True)
    inep_code = Column(String(8), unique=True, index=True)
    name = Column(String(100))
    address = Column(String(50))
    neighborhood = Column(String(50))
    cep = Column(String(8))
    phone = Column(String(20))
    principal = Column(String(50))
    email = Column(String(50))
    login = Column(String(20), unique=True)
    password_hash = Column(String(255))
    config = relationship("SimulationConfig", back_populates="school", uselist=False, cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(20))
    question_text = Column(Text)
    option_a = Column(String(500))
    option_b = Column(String(500))
    option_c = Column(String(500))
    option_d = Column(String(500))
    option_e = Column(String(500), nullable=True)
    correct_answer = Column(String(1))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime, nullable=True)


class SimulationConfig(Base):
    __tablename__ = "simulation_configs"
    id = Column(Integer, primary_key=True, index=True)
    school_id = Column(Integer, ForeignKey("schools.id"))
    portuguese_questions = Column(Integer, default=10)
    math_questions = Column(Integer, default=10)
    portuguese_time = Column(Integer, default=50)
    math_time = Column(Integer, default=50)
    num_options = Column(Integer, default=4)
    avg_time_per_question = Column(Integer, default=3)
    time_suggestion = Column(String(200), default="Sugestão: Passe para a próxima pergunta! Depois você volta.")
    show_final_score = Column(Boolean, default=True)
    save_student_info = Column(Boolean, default=True)
    simulation_link = Column(String(100), unique=True, index=True)
    school = relationship("School", back_populates="config")

class SimulationResult(Base):
    __tablename__ = "simulation_results"
    id = Column(Integer, primary_key=True, index=True)
    school_id = Column(Integer, ForeignKey("schools.id"))
    student_name = Column(String(100))
    portuguese_score = Column(Integer, default=0)
    math_score = Column(Integer, default=0)
    portuguese_total = Column(Integer)
    math_total = Column(Integer)
    completion_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    school = relationship("School")

# #############################################################
# ALTERAÇÃO 1: A LINHA ABAIXO FOI REMOVIDA DAQUI
# Base.metadata.create_all(bind=engine)
# O ARQUIVO init_db.py AGORA É RESPONSÁVEL POR ISSO.
# #############################################################

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="Sistema SAEB", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Funções Auxiliares e Middleware ---
def hash_password(password: str) -> str: return hashlib.sha256(password.encode()).hexdigest()
def verify_password(password: str, hashed: str) -> bool: return hash_password(password) == hashed

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)




@app.get("/init-database")
async def initialize_database():
    # Só permite executar uma vez ou em desenvolvimento
    if os.environ.get("ALLOW_DB_INIT") == "true":
        exec(open('init_db.py').read())
        return {"message": "Database initialized successfully"}
    return {"message": "Database initialization not allowed"}



def get_current_user(access_token: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    if access_token is None: return None
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id, user_type = payload.get("sub"), payload.get("user_type")
        if user_type == "superuser": return {"user": db.query(Superuser).get(user_id), "type": "superuser"}
        if user_type == "school": return {"user": db.query(School).get(user_id), "type": "school"}
    except jwt.PyJWTError: return None
    return None

@app.middleware("http")
async def add_user_to_state(request: Request, call_next):
    db = SessionLocal()
    try:
        request.state.user = get_current_user(request.cookies.get("access_token"), db)
    finally:
        db.close()
    response = await call_next(request)
    return response

# #############################################################
# ALTERAÇÃO 2: A FUNÇÃO DE STARTUP FOI REMOVIDA
# O ARQUIVO init_db.py AGORA É RESPONSÁVEL PELA INICIALIZAÇÃO.
# @app.on_event("startup")
# def startup_event():
#     ...
# #############################################################


# --- Rotas Gerais e de Autenticação ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Página Inicial", "error": error})

# ... (O RESTO DO SEU CÓDIGO DE ROTAS CONTINUA EXATAMENTE O MESMO, SEM NENHUMA ALTERAÇÃO)

@app.post("/start-simulation-by-code")
async def start_simulation_by_code(request: Request, school_code: str = Form(...), db: Session = Depends(get_db)):
    school = db.query(School).filter(School.inep_code == school_code).first()
    if not school or not school.config:
        return RedirectResponse(url="/?error=Código da escola inválido ou não encontrado.", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=f"/simulado/{school.config.simulation_link}", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "title": "Login"})

@app.post("/login")
async def login(request: Request, db: Session = Depends(get_db), username: str = Form(...), password: str = Form(...)):
    user = db.query(Superuser).filter_by(username=username).first()
    if user and verify_password(password, user.password_hash):
        token = create_access_token({"sub": str(user.id), "user_type": "superuser"})
        response = RedirectResponse("/admin/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie("access_token", token, httponly=True)
        return response
    school = db.query(School).filter_by(login=username).first()
    if school and verify_password(password, school.password_hash):
        token = create_access_token({"sub": str(school.id), "user_type": "school"})
        response = RedirectResponse("/school/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie("access_token", token, httponly=True)
        return response
    return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciais inválidas"})

@app.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

@app.get("/simulado/cancel")
async def cancel_simulation():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("student_name")
    response.delete_cookie("simulation_data")
    return response


# --- Rotas do Superusuário ---
def require_superuser(current_user: dict = Depends(get_current_user)):
    if not current_user or current_user["type"] != "superuser":
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})

@app.get("/admin/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_superuser)])
async def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    schools = db.query(School).order_by(School.name).all()
    questions_count = { "portugues": db.query(Question).filter_by(subject="portugues").count(), "matematica": db.query(Question).filter_by(subject="matematica").count() }
    return templates.TemplateResponse("admin_dashboard.html", {"request": request, "schools": schools, "questions_count": questions_count, "title": "Dashboard Admin"})

@app.post("/admin/schools/add", dependencies=[Depends(require_superuser)])
async def add_school(db: Session = Depends(get_db), name: str = Form(...), inep_code: str = Form(...), address: str = Form(...), neighborhood: str = Form(...), cep: str = Form(...), phone: str = Form(...), principal: str = Form(...), email: str = Form(...), login: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password: return RedirectResponse("/admin/dashboard?error=Senhas não coincidem", status_code=status.HTTP_302_FOUND)
    new_school = School(name=name, inep_code=inep_code, address=address, neighborhood=neighborhood, cep=cep, phone=phone, principal=principal, email=email, login=login, password_hash=hash_password(password))
    db.add(new_school)
    db.commit(); db.refresh(new_school)
    db.add(SimulationConfig(school_id=new_school.id, simulation_link=secrets.token_urlsafe(16)))
    db.commit()
    return RedirectResponse("/admin/dashboard?success=Escola cadastrada com sucesso!", status_code=status.HTTP_302_FOUND)

@app.get("/admin/school/form/{school_id}", response_class=HTMLResponse, dependencies=[Depends(require_superuser)])
async def get_school_form(request: Request, school_id: int, db: Session = Depends(get_db)):
    school = db.query(School).get(school_id)
    if not school: raise HTTPException(status_code=404, detail="Escola não encontrada")
    return templates.TemplateResponse("admin_school_form.html", {"request": request, "school": school, "title": "Alterar Dados da Escola"})

@app.post("/admin/school/update/{school_id}", dependencies=[Depends(require_superuser)])
async def update_school(school_id: int, db: Session = Depends(get_db), name: str = Form(...), inep_code: str = Form(...), address: str = Form(...), neighborhood: str = Form(...), cep: str = Form(...), phone: str = Form(...), principal: str = Form(...), email: str = Form(...), login: str = Form(...), new_password: Optional[str] = Form(None)):
    school = db.query(School).get(school_id)
    if not school: raise HTTPException(status_code=404, detail="Escola não encontrada")
    school.name, school.inep_code, school.address, school.neighborhood, school.cep, school.phone, school.principal, school.email, school.login = name, inep_code, address, neighborhood, cep, phone, principal, email, login
    if new_password: school.password_hash = hash_password(new_password)
    db.commit()
    return RedirectResponse("/admin/dashboard?success=Informações da escola salvas com sucesso!", status_code=status.HTTP_302_FOUND)

@app.post("/admin/schools/delete/{school_id}", dependencies=[Depends(require_superuser)])
async def delete_school(school_id: int, db: Session = Depends(get_db)):
    school = db.query(School).get(school_id)
    if school: 
        db.delete(school)
        db.commit()
    return RedirectResponse("/admin/dashboard?success=Escola excluída com sucesso!", status_code=status.HTTP_302_FOUND)

@app.get("/admin/questions", response_class=HTMLResponse, dependencies=[Depends(require_superuser)])
async def admin_questions(request: Request, subject: str, search: Optional[str] = Query(None), page: int = Query(1, ge=1), db: Session = Depends(get_db)):
    query = db.query(Question).filter_by(subject=subject)
    if search: 
        query = query.filter(Question.question_text.contains(search))
    total_items = query.count()
    total_pages = math.ceil(total_items / PAGE_SIZE)
    offset = (page - 1) * PAGE_SIZE
    questions = query.order_by(desc(Question.created_at)).offset(offset).limit(PAGE_SIZE).all()
    return templates.TemplateResponse("admin_questions.html", { "request": request, "questions": questions, "subject": subject, "search": search or "", "current_page": page, "total_pages": total_pages, "total_items": total_items, "title": f"Questões de {subject.capitalize()}" })

@app.get("/admin/question/form", response_class=HTMLResponse, dependencies=[Depends(require_superuser)])
async def question_form(request: Request, subject: str, id: Optional[int] = Query(None), db: Session = Depends(get_db)):
    question = db.query(Question).get(id) if id else None
    title = "Editar Questão" if id else "Nova Questão"
    return templates.TemplateResponse("admin_question_form.html", {"request": request, "subject": subject, "question": question, "title": title})

@app.post("/admin/question/save", dependencies=[Depends(require_superuser)])
async def save_question(db: Session = Depends(get_db), id: Optional[int] = Query(None), subject: str = Form(...), question_text: str = Form(...), option_a: str = Form(...), option_b: str = Form(...), option_c: str = Form(...), option_d: str = Form(...), option_e: Optional[str] = Form(None), correct_answer: str = Form(...)):
    if id:
        q = db.query(Question).get(id)
    else:
        q = Question(subject=subject)
        db.add(q)
    q.question_text, q.option_a, q.option_b, q.option_c, q.option_d, q.option_e, q.correct_answer = question_text, option_a, option_b, option_c, option_d, option_e, correct_answer.upper()
    db.commit()
    return RedirectResponse(f"/admin/questions?subject={subject}", status_code=status.HTTP_302_FOUND)

@app.post("/admin/question/delete/{question_id}", dependencies=[Depends(require_superuser)])
async def delete_question(question_id: int, db: Session = Depends(get_db)):
    q = db.query(Question).get(question_id)
    if q: 
        subject = q.subject
        db.delete(q)
        db.commit()
        return RedirectResponse(f"/admin/questions?subject={subject}", status_code=status.HTTP_302_FOUND)
    raise HTTPException(status_code=404)

@app.post("/admin/questions/shuffle", dependencies=[Depends(require_superuser)])
async def shuffle_questions_answers(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    question_ids = data.get("question_ids")
    if not question_ids:
        return JSONResponse(status_code=400, content={"message": "Nenhum ID de questão fornecido."})
    questions = db.query(Question).filter(Question.id.in_(question_ids)).all()
    for q in questions:
        correct_text = getattr(q, f"option_{q.correct_answer.lower()}")
        options = []
        if q.option_a: options.append(q.option_a)
        if q.option_b: options.append(q.option_b)
        if q.option_c: options.append(q.option_c)
        if q.option_d: options.append(q.option_d)
        if q.option_e: options.append(q.option_e)
        random.shuffle(options)
        q.option_a, q.option_b, q.option_c, q.option_d, q.option_e = None, None, None, None, None
        new_correct_letter = ""
        letters = ['a', 'b', 'c', 'd', 'e']
        for i, text in enumerate(options):
            setattr(q, f"option_{letters[i]}", text)
            if text == correct_text:
                new_correct_letter = letters[i].upper()
        q.correct_answer = new_correct_letter
    db.commit()
    return JSONResponse(content={"message": f"{len(questions)} questões foram embaralhadas com sucesso!"})


# --- Rotas da Escola ---
def require_school(current_user: dict = Depends(get_current_user)):
    if not current_user or current_user["type"] != "school":
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})

@app.get("/school/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_school)])
async def school_dashboard(request: Request, filter_name: Optional[str] = Query(None), current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    school = current_user["user"]
    query = db.query(SimulationResult).filter_by(school_id=school.id)
    if filter_name: query = query.filter(SimulationResult.student_name.contains(filter_name))
    return templates.TemplateResponse("school_dashboard.html", {"request": request, "school": school, "config": school.config, "results": query.order_by(desc(SimulationResult.completion_time)).all(), "title": "Painel da Escola", "filter_name": filter_name or ""})

@app.post("/school/config/update", dependencies=[Depends(require_school)])
async def update_config(db: Session = Depends(get_db), current_user=Depends(get_current_user), 
                        portuguese_questions: int = Form(...), math_questions: int = Form(...), 
                        portuguese_time: int = Form(...), math_time: int = Form(...), 
                        num_options: int = Form(...), avg_time_per_question: int = Form(...), 
                        time_suggestion: str = Form(...), show_final_score: bool = Form(False),
                        save_student_info: bool = Form(False)):
    config = db.query(SimulationConfig).filter_by(school_id=current_user["user"].id).first()
    if config:
        config.portuguese_questions, config.math_questions, config.portuguese_time, config.math_time, config.num_options, config.avg_time_per_question, config.time_suggestion, config.show_final_score, config.save_student_info = portuguese_questions, math_questions, portuguese_time, math_time, num_options, avg_time_per_question, time_suggestion, show_final_score, save_student_info
        db.commit()
        return JSONResponse({"message": "Configurações salvas com sucesso!"})
    return JSONResponse({"message": "Erro ao salvar."}, status_code=400)


# --- ROTAS DO SIMULADO ---
@app.get("/simulado/{link_id}", response_class=HTMLResponse)
async def simulation_entry(request: Request, link_id: str, db: Session = Depends(get_db)):
    config = db.query(SimulationConfig).filter_by(simulation_link=link_id).first()
    if not config: raise HTTPException(404, "Simulado não encontrado")
    return templates.TemplateResponse("simulation_start.html", {"request": request, "school": config.school, "link_id": link_id, "title": "Iniciar Simulado"})

@app.post("/simulado/{link_id}/start")
async def simulation_start(link_id: str, student_name: str = Form(...), db: Session = Depends(get_db)):
    config = db.query(SimulationConfig).filter_by(simulation_link=link_id).first()
    if not config: raise HTTPException(404)

    def select_questions(subject: str, count: int):
        query = db.query(Question).filter_by(subject=subject).order_by(
            Question.last_used_at.is_(None).desc(),
            Question.last_used_at.asc(),
            Question.created_at.desc()
        )
        selected = query.limit(count).all()
        now = datetime.now(timezone.utc)
        for q in selected:
            q.last_used_at = now
        db.commit()
        random.shuffle(selected)
        return [q.id for q in selected]

    questions_p = select_questions("portugues", config.portuguese_questions)
    questions_m = select_questions("matematica", config.math_questions)
    
    if not questions_p and not questions_m: 
        raise HTTPException(500, "Não há questões cadastradas para este simulado.")
    
    now_timestamp = datetime.now(timezone.utc).timestamp()
    simulation_data = {"p_ids": questions_p, "m_ids": questions_m, "answers": {}, "start_time_p": now_timestamp, "start_time_m": None}
    
    response = RedirectResponse(f"/simulado/{link_id}/question/1", status_code=status.HTTP_302_FOUND)
    response.set_cookie("student_name", student_name)
    response.set_cookie("simulation_data", json.dumps(simulation_data))
    return response

@app.get("/simulado/{link_id}/question/{q_idx}", response_class=HTMLResponse)
async def simulation_question(request: Request, response: Response, link_id: str, q_idx: int, student_name: Optional[str] = Cookie(None), simulation_data: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    if not all([student_name, simulation_data]): return RedirectResponse(f"/simulado/{link_id}")
    config = db.query(SimulationConfig).filter_by(simulation_link=link_id).first()
    sim_data = json.loads(simulation_data)
    p_ids, m_ids = sim_data["p_ids"], sim_data["m_ids"]
    is_portuguese = q_idx <= len(p_ids)
    subject = "portugues" if is_portuguese else "matematica"
    
    now_ts = datetime.now(timezone.utc).timestamp()
    
    if "general_start_time" not in sim_data:
        sim_data["general_start_time"] = sim_data["start_time_p"]
        response.set_cookie("simulation_data", json.dumps(sim_data))
    
    total_general_time = (config.portuguese_time + config.math_time) * 60
    general_elapsed = now_ts - sim_data["general_start_time"]
    general_time_left = max(0, total_general_time - general_elapsed)
    
    if not is_portuguese and sim_data.get("start_time_m") is None:
        sim_data["start_time_m"] = now_ts
        response.set_cookie("simulation_data", json.dumps(sim_data))

    question_time_key = f"question_start_{q_idx}"
    if question_time_key not in sim_data:
        sim_data[question_time_key] = now_ts
        response.set_cookie("simulation_data", json.dumps(sim_data))
    
    question_time_limit = config.avg_time_per_question * 60
    question_elapsed = now_ts - sim_data[question_time_key]
    question_time_left = max(0, question_time_limit - question_elapsed)
    
    question_id = p_ids[q_idx - 1] if is_portuguese else m_ids[q_idx - len(p_ids) - 1]
    question = db.query(Question).get(question_id)
    total_questions = len(p_ids) + len(m_ids)
    options = [("A", question.option_a), ("B", question.option_b), ("C", question.option_c), ("D", question.option_d), ("E", question.option_e)]
    visible_options = [opt for opt in options if opt[1] is not None][:config.num_options]
    
    return templates.TemplateResponse("simulation_question.html", {
        "request": request, "config": config, "question": question, "options": visible_options, "subject": subject, 
        "current_q_idx": q_idx, "total_q": total_questions, "student_name": student_name, "link_id": link_id, 
        "sim_data": sim_data, "general_time_left": int(general_time_left), "question_time_left": int(question_time_left),
        "title": f"Questão {q_idx}"
    })

@app.post("/simulado/{link_id}/navigate")
async def simulation_navigate(response: Response, link_id: str, q_idx: int = Form(...), question_id: int = Form(...), answer: Optional[str] = Form(None), action: str = Form(...), simulation_data: Optional[str] = Cookie(None)):
    sim_data = json.loads(simulation_data)
    if answer:
        sim_data["answers"][str(question_id)] = answer
    next_q_idx = q_idx
    if action == "next": next_q_idx += 1
    elif action == "previous": next_q_idx -= 1
    total_questions = len(sim_data["p_ids"]) + len(sim_data["m_ids"])
    if action == "finish" or next_q_idx > total_questions:
        next_url = f"/simulado/{link_id}/finish"
    else:
        next_url = f"/simulado/{link_id}/question/{next_q_idx}"
    res = RedirectResponse(next_url, status_code=status.HTTP_302_FOUND)
    res.set_cookie("simulation_data", json.dumps(sim_data))
    return res

@app.get("/simulado/{link_id}/finish")
async def simulation_finish(request: Request, link_id: str, student_name: Optional[str] = Cookie(None), simulation_data: Optional[str] = Cookie(None), db: Session = Depends(get_db)):
    if not all([student_name, simulation_data]): return RedirectResponse(f"/simulado/{link_id}")
    config = db.query(SimulationConfig).filter_by(simulation_link=link_id).first()
    sim_data = json.loads(simulation_data)
    answers, p_ids, m_ids = sim_data["answers"], sim_data["p_ids"], sim_data["m_ids"]
    p_score, m_score = 0, 0
    
    all_question_ids = p_ids + m_ids
    questions = db.query(Question).filter(Question.id.in_(all_question_ids)).all()
    question_map = {q.id: q for q in questions}
    
    wrong_questions = []
    for q_id in all_question_ids:
        question = question_map.get(q_id)
        if not question: continue
            
        user_answer = answers.get(str(q_id))
        is_correct = user_answer == question.correct_answer
        
        if is_correct:
            if q_id in p_ids: p_score += 1
            elif q_id in m_ids: m_score += 1
        else:
            question_data = {
                'id': question.id, 'subject': question.subject, 'question_text': question.question_text,
                'option_a': question.option_a, 'option_b': question.option_b, 'option_c': question.option_c,
                'option_d': question.option_d, 'option_e': question.option_e, 'correct_answer': question.correct_answer,
                'user_answer': user_answer
            }
            wrong_questions.append(question_data)
    
    result_data = { 
        "student_name": student_name, "portuguese_score": p_score, "math_score": m_score, 
        "portuguese_total": len(p_ids), "math_total": len(m_ids) 
    }
    
    if config.save_student_info:
        result_to_save = SimulationResult(**result_data, school_id=config.school_id)
        db.add(result_to_save)
        db.commit()
    
    response = templates.TemplateResponse("simulation_result.html", { 
        "request": request, "student_name": student_name, "result": result_data, 
        "wrong_questions": wrong_questions, "show_score": config.show_final_score, 
        "title": "Fim do Simulado" 
    })
    response.delete_cookie("student_name")
    response.delete_cookie("simulation_data")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)