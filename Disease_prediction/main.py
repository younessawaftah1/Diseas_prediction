from flask import Flask, flash 
from flask import render_template
from flask import request
import numpy as np 
import pandas as pd
import pickle 

app =Flask(__name__)
model=pickle.load(open('Disease.pkl','rb'))
app.config['SECRET_KEY']='thisissecret'

@app.route("/home")
def home():
	return render_template("testHtml.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
	if request.method == 'POST':
		q1=int(request.form['q1'])
		q2=int(request.form['q2'])
		q3=int(request.form['q3'])
		q4=int(request.form['q4'])
		q5=int(request.form['q5'])
		q6=int(request.form['q6'])
		q7=int(request.form['q7'])
		q8=int(request.form['q8'])
		q9=int(request.form['q9'])
		q10=int(request.form['q10'])
		q11=int(request.form['q11'])
		q12=int(request.form['q12'])
		q13=int(request.form['q13'])
		q14=int(request.form['q14'])
		q15=int(request.form['q15'])
		q16=int(request.form['q16'])		
		q17=int(request.form['q17'])
		q18=int(request.form['q18'])
		q19=int(request.form['q19'])
		q20=int(request.form['q20'])
		q21=int(request.form['q21'])
		q22=int(request.form['q22'])
		q23=int(request.form['q23'])
		q24=int(request.form['q24'])
		q25=int(request.form['q25'])
		q26=int(request.form['q26'])
		q27=int(request.form['q27'])
		a=np.array([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13
			,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23
		,q24,q25
		,q26,q27
		]])		
		result=model.predict(a)
		flash("The Disease what our model predicted is: ",result)
		return render_template("testHtml.html",result=result)
        
if __name__=="__main__":
	app.run(debug=True)
