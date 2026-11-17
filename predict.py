import pandas as pd
import joblib

print("\nEnter student information")

student = {}

student["age"] = int(input("Age: "))

student["Medu"] = int(input("Mother education (0-4): "))
student["Fedu"] = int(input("Father education (0-4): "))

student["studytime"] = int(input("Study time (1-4): "))
student["failures"] = int(input("Past failures (0-3): "))

student["schoolsup"] = input("School support (yes/no): ")
student["famsup"] = input("Family support (yes/no): ")
student["higher"] = input("Wants higher education (yes/no): ")
student["internet"] = input("Internet access (yes/no): ")

student["famrel"] = int(input("Family relationship quality (1-5): "))
student["freetime"] = int(input("Free time (1-5): "))
student["goout"] = int(input("Going out (1-5): "))
student["Dalc"] = int(input("Workday alcohol consumption (1-5): "))
student["health"] = int(input("Health (1-5): "))
student["absences"] = int(input("Absences: "))

student["G1"] = int(input("First period grade G1 (0-20): "))
student["G2"] = int(input("Second period grade G2 (0-20): "))

student_df = pd.DataFrame([student])

model = joblib.load("student_marks_model.pk1")

prediction = model.predict(student_df)[0]

prediction = max(0, min(20, prediction))

print("\n==============================")
print("PREDICTED FINAL GRADE")
print("==============================")
print(f"G3: {prediction:.2f} / 20")