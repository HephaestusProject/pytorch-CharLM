FROM python:3.7-stretch@sha256:ba2b519dbdacc440dd66a797d3dfcfda6b107124fa946119d45b93fc8f8a8d77

WORKDIR /app

RUN apt-get clean \
    && apt-get -y update

RUN pip install --upgrade pip
RUN pip install flask==1.1.1
RUN pip install flask-restplus==0.12.1
RUN pip install pytest
RUN pip install pyyaml

COPY . .

CMD [ "python", "./api.py" ]