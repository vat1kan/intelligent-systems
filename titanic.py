import numpy as np
import pandas as pd
    
# open csv file and create data frame 
titanic = pd.read_csv('titanic.csv')
df = pd.DataFrame(titanic)

#0 - male; 1 - female
#0 - not survived; 1 - survived
#group by attribure 'sex' and 'survived'
sex_sample = df.groupby(['Sex','Survived'])['PassengerID'].count()

#print table which shows the number of survivors of each sex
print("\nSex: 0 - male, 1 - female ; Survived: 0 - not survived, 1 - survived\n")
print(sex_sample,'\n')

#calculate the ratio of surviving males relative to all passengers
male_rel = np.round((sex_sample[0,1])/len(df),3)
#print(f"{sex_sample[0,1]} / {len(df)}={male_rel}")

#calculate the ratio of surviving females relative to all passengers
fem_rel = np.round((sex_sample[1,1])/len(df),3)
#print(f"{sex_sample[1,1]} / {len(df)}={fem_rel}")

#compare relations and accept the meaning of gender
if male_rel > fem_rel:
    print(f"The male sex had a greater chance of being saved. The ratio of surviving men to the total number of passengers: {male_rel}")
else:
    print(f"The female sex had a greater chance of being saved. The ratio of surviving women to the total number of passengers: {fem_rel}")

#group the values ​​of the age parameter with a gradation of 10 years
age_sample = pd.cut(df['Age'],bins=10)

#determine the number of survivors in each age category
age_table = pd.crosstab(age_sample,df['Survived'])

#print the category with a greater chance of being saved
print(f"\nThe age category had a greater change of being saved:{age_table.idxmax()[1]}")
