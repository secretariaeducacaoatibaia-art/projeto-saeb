# ======== Conteúdo do arquivo init_db.py ========

print("Iniciando a criação do banco de dados...")

try:
    # Importa as ferramentas do seu código
    from main import Base, engine, SessionLocal, Superuser, Question, hash_password
    from sample_questions import portuguese_questions, math_questions

    # Cria as tabelas (se não existirem)
    Base.metadata.create_all(bind=engine)
    print("Tabelas verificadas/criadas.")

    # Inicia uma sessão com o banco
    db = SessionLocal()

    # Adiciona o superusuário (se não existir)
    if not db.query(Superuser).filter_by(username="admin").first():
        db.add(Superuser(username="admin", password_hash=hash_password("admin123")))
        db.commit()
        print("Superusuário 'admin' criado.")
    else:
        print("Superusuário 'admin' já existe.")

    # Adiciona as questões (se não existirem)
    if db.query(Question).count() == 0:
        print("Populando o banco com questões de exemplo...")
        for q_data in portuguese_questions + math_questions:
            db.add(Question(**q_data))
        db.commit()
        print(f"{len(portuguese_questions) + len(math_questions)} questões adicionadas.")
    else:
        print("Questões já existem no banco de dados. Nenhuma ação necessária.")

    db.close()
    print("Operação concluída com sucesso!")

except Exception as e:
    print(f"Ocorreu um erro: {e}")