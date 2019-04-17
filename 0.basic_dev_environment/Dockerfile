FROM alpine:3.6

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/main | tee /etc/apk/repositories \
    && echo http://dl-cdn.alpinelinux.org/alpine/edge/testing | tee -a /etc/apk/repositories \
    && echo http://dl-cdn.alpinelinux.org/alpine/edge/community | tee -a /etc/apk/repositories \
    && echo "|--> Install basics pre-requisites" \
    && apk add -U --no-cache tini bash \
        curl ca-certificates python3 py3-numpy py3-numpy-f2py \
        freetype jpeg libpng libstdc++ libgomp graphviz font-noto \
## Setup de basic requeriments
    && echo "|--> Install Python basics" \
    && python3 -m ensurepip \
    && rm -r /usr/lib/python*/ensurepip \
    && pip3 --no-cache-dir install --upgrade pip setuptools wheel \
    && if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip; fi \
    && if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi \
    && ln -s locale.h /usr/include/xlocale.h \
## Dev dependencies and others stuffs...
    && echo "|--> Install build dependencies" \
    && apk add -U --no-cache --virtual=.build-deps \
        build-base linux-headers python3-dev git cmake jpeg-dev \
        libffi-dev openblas-dev py-numpy-dev freetype-dev libpng-dev \
    && echo "|--> Install Python packages" \
    && pip install -U --no-cache-dir pyyaml pymkl cffi scikit-learn \
        matplotlib ipywidgets notebook requests pillow pandas flask\
## Cleaning
    && echo "|--> Cleaning" \
    && rm /usr/include/xlocale.h \
    && rm -rf /root/.cache \
    && rm -rf /var/cache/apk/* \
    && apk del .build-deps \
    && find /usr/lib/python3.6 -name __pycache__ | xargs rm -r \
    && rm -rf /root/.[acpw]* \
## Run notebook without token and disable warnings
    && echo "|--> Configure Jupyter extension" \
    && jupyter nbextension enable --py widgetsnbextension \
    && mkdir -p ~/.ipython/profile_default/startup/ \
    && echo "import warnings" >> ~/.ipython/profile_default/startup/config.py \
    && echo "warnings.filterwarnings('ignore')" >> ~/.ipython/profile_default/startup/config.py \
    && echo "c.NotebookApp.token = u''" >> ~/.ipython/profile_default/startup/config.py \
    && echo "|--> Done!"

EXPOSE 8888

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/main | tee /etc/apk/repositories \
    && echo http://dl-cdn.alpinelinux.org/alpine/edge/testing | tee -a /etc/apk/repositories \
    && echo http://dl-cdn.alpinelinux.org/alpine/edge/community | tee -a /etc/apk/repositories \
    && echo "|--> Install basics pre-requisites" \
    && apk add -U --no-cache git
RUN pip install gitpython
WORKDIR /notebooks

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
    "--allow-root", "--ip=0.0.0.0", "--NotebookApp.token="]
