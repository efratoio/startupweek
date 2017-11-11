<<<<<<< HEAD
FROM python:3.5.2

# set working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# add requirements (to leverage Docker cache)
ADD ./requirements.txt /usr/src/app/requirements.txt

# install requirements
RUN pip install -r requirements.txt
=======
FROM node:latest

# set working directory
RUN mkdir /usr/src/app
WORKDIR /usr/src/app

# add `/usr/src/app/node_modules/.bin` to $PATH
ENV PATH /usr/src/app/node_modules/.bin:$PATH

# install and cache app dependencies
ADD package.json /usr/src/app/package.json
RUN npm install --silent
RUN npm install react-scripts@0.9.5 -g --silent
>>>>>>> 3dbad9b7b314d507c02890d70e45c16804793871

# add app
ADD . /usr/src/app

<<<<<<< HEAD
# run server
CMD python manager.py runserver -h 0.0.0.0
=======
# start app
CMD ["npm", "start"]
>>>>>>> 3dbad9b7b314d507c02890d70e45c16804793871
