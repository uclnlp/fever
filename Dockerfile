FROM tensorflow/tensorflow:1.8.0--gpu-py3

# install necessary programs
RUN apt-get update && \
  apt-get install -y locales wget git

# Set the locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

VOLUME /home/hexaf
VOLUME /home/hexaf/fever
