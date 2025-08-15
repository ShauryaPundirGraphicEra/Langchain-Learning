from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    #  name:str= "XYZ"     #default value for name
    age:Optional[int]=None
    email:EmailStr          # email validation
    cgpa : float = Field(gt=0,lt=10,default=5,description="The decimal value representating the cgpa of a student")
    
    # descryption working as Annotation of type_dict
    
    
    

# new_student={ "name": "Shaurya Pundir"}

# new_student={ "name": "Shaurya Pundir","age":20}

# new_student={ "name": "Shaurya Pundir","age": "20" }  // type coercing ,means that pydantic automatically converts the string "20" to integer 20

new_student={ "name": "Shaurya Pundir","age":20, "email":"abc1435@gmail.com",'cgpa':'8.25'}

student = Student(**new_student)

print(student)

# student_dict= dict(student)

# student_json= model_dump_json()
