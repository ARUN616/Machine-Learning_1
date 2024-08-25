from flask import Flask
app=Flask(__name__)

# URL Binding
@app.route('/')
def hello():
    return("I am Arun!!")

@app.route('/page2')
def page2():
    return("I am Page2")

@app.route('/admin')
def admin():
    return("I am Admin")
app.run(port=8000)## Changing the port number and debug has turned on and off



