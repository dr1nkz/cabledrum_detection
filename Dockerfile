FROM python:3

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install libgl1 -y

CMD [ "python", "test.py" ]
