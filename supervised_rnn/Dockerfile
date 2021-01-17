FROM python:3.6

MAINTAINER Sam Rahbar

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY RNN /opt/program
WORKDIR /opt/program
RUN python3 train.py