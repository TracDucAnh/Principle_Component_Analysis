import os #Thư viện làm việc với hệ điều hành
import numpy as np
import pandas as pd #Thư viện làm việc với dữ liệu
import matplotlib.pyplot as plt #Thư viện trực quan hóa dữ liệu
from sklearn.decomposition import PCA #Thư viện tích hợp sẵn PCA

current_path = os.getcwd() #Lấy đường dẫn ở thư mục làm việc hiện tại

print("\nCurrent location:",current_path,"\n") #In đường dẫn thư mục hiện tại

data_path = os.path.join(current_path, "dataset", "heart_disease_health_indicators_BRFSS2015.csv") #Đường dẫn tới file dữ liệu

print("File data location:",data_path, "\n") #In đường dẫn tới file dữ liệu

data = pd.read_csv(data_path) #Đọc file dữ liệu, lưu vào biến data, là một Dataframe

print("Header:",data.columns) #In ra tất cả các chiều của dữ liệu

print("\n 5 rows from dataset:")

print(data.head()) #In 5 dòng đầu trong file dữ liệu

data = data.drop("HeartDiseaseorAttack", axis = 1)

pca = PCA(n_components=5)

principle_component = pca.fit_transform(data)

plt.plot(np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel('(Thành Phần Chính) Principal Component')
plt.ylabel('(Tỷ lệ phương sai được giải thích) Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()