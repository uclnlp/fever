FROM continuumio/miniconda3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
  zip \
  gzip \
  make \
  automake \
  gcc \
  build-essential \
  g++ \
  cpp \
  libc6-dev \
  man-db \
  autoconf \
  pkg-config \
  unzip \
  libffi-dev \
  software-properties-common \
  locales \
  wget \
  git \
  python-dev

# Set the locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir -pv /hexaf/fever
RUN mkdir -pv /hexaf/fever/results
WORKDIR /hexaf/fever

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN conda install -y -c conda-forge spacy==1.9.0

# download model and index files
ADD http://tti-coin.jp/data/yoneda/fever/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip /tmp/
ADD http://tti-coin.jp/data/yoneda/fever/data.zip /tmp/
ADD initial_setup_fever2.sh .
RUN unzip /tmp/data.zip -d .
RUN unzip /tmp/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission.zip -d /hexaf/fever/results

RUN chmod u+x initial_setup_fever2.sh && /hexaf/fever/initial_setup_fever2.sh
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('gazetteers'); nltk.download('names')"

# add config files
ADD configs/submission_config.json configs/
ADD configs/base_config.json configs/
ADD pipeline.py .
ADD setup.sh .
RUN chmod u+x setup.sh && /hexaf/fever/setup.sh

RUN mkdir -pv src
ADD src src
ADD predict.sh .

ENV PYTHONPATH /hexaf/fever/:/hexaf/jack/:src
ENV FLASK_APP app:hexaf_fever

#ENTRYPOINT ["/bin/bash","-c"]
# CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "app:hexaf_fever"]
CMD ["python" "src/waitress_san.py"]
