
from flask import Flask, request, render_template, redirect, url_for
import finalMl as ml

app = Flask(__name__, template_folder='templates')



@app.route('/', methods=["POST", "GET"])
def register():
    if request.method == "POST":
        li0 = int(request.form['fd1'])
        li1 = int(request.form['fd2'])
        income1 = int(request.form['fd3income'])
        income2 = int(request.form['fd4income'])
        li3 = int(request.form['td1'])
        li4 = int(request.form['td2'])
        
        est = ml.predict(li0, li1, income1, income2, li3, li4)
    
        return render_template("display.html", h=est)
    else:
        return render_template("register.html")



if __name__ == "__main__":
    app.run(debug=True)