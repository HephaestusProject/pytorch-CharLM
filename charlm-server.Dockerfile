FROM python:3.7-stretch@sha256:ba2b519dbdacc440dd66a797d3dfcfda6b107124fa946119d45b93fc8f8a8d77

WORKDIR /app

RUN apt-get clean \
    && apt-get -y update

RUN pip install --upgrade pip
RUN pip install pytest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD [ "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]