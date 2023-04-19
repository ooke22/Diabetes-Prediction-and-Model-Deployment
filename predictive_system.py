import numpy as np
import pickle
#loading the saved model
model = pickle.load(open('C:/Diabetes_ML_Deployment/diabetes_model.pkl', 'rb'))

new_data = (5, 166,72, 19, 175, 26, 51)
#changing the new_data to numpy array
new_data_as_numpy_array = np.asarray(new_data)

#reshape the array as we are predicting for one instance
new_data_reshaped = new_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(new_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This individuals is not diabetic')
else:
    print('This individual is diabetic')



#%%
