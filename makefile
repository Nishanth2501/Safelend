# Makefile for SafeLend

# Download / place data manually into data/raw/
# These targets assume you have requirements installed

prepare-data:
	@echo "1) Python cleaning → interim clean parquet"
	python3 src/data/preprocess.py
	@echo "2) SQL aggregations → interim *_agg.parquet"
	python3 src/data/run_sql_aggregrations.py
	@echo "3) Build processed train/test tables"
	python3	 src/data/build_processed.py
	@echo "4) Run sanity checks"
	python3 src/data/sanity_checks.py
post-clean:
	@echo "Running post-sanity cleanup..."
	python3 src/data/post_sanity_cleanup.py

sanity-check:
	@echo "Running data sanity checks..."
	python3 src/data/sanity_checks.py

eda:
	@echo "Running exploratory data analysis notebook..."
	jupyter nbconvert --to html --execute notebooks/01_eda.ipynb --output notebooks/eda_report.html

train:
	@echo "Training model..."
	python3 src/models/train.py

evaluate:
	@echo "Evaluating model..."
	python3 src/models/evaluate.py

serve:
	@echo "Starting FastAPI service..."
	uvicorn src.service.server:app --reload --host 0.0.0.0 --port 8000

ui:
	@echo "Starting SafeLend UI..."
	@echo "Open your browser to: http://localhost:8000"
	uvicorn src.service.server:app --reload --host 0.0.0.0 --port 8000

demo:
	@echo "Starting SafeLend Demo..."
	@echo "Open your browser to: http://localhost:8080/demo.html"
	python3 serve_demo.py

predict:
	@echo "Starting SafeLend Prediction System..."
	@echo "Open your browser to: http://localhost:8000"
	python3 final_predict_server.py

interactive:
	@echo "Starting SafeLend Interactive Demo..."
	@echo "Enter loan application details to get AI predictions"
	python3 prediction_demo.py

docker-build:
	@echo "Building Docker image..."
	docker build -t safelend:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 safelend:latest

test:
	@echo "Running tests..."
	pytest tests/ -v