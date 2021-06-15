FROM python:3.9.5

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD python city_analysis.py && python reviews_analysis.py