FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
WORKDIR /quora-question-pairs

COPY model /quora-question-pairs/model
COPY app /quora-question-pairs/app
COPY main.py /quora-question-pairs
COPY requirements.txt /quora-question-pairs

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:80