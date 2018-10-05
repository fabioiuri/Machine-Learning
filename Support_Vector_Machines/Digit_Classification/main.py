import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

def view_img(i):
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img)
    plt.title(train_labels.iloc[i,0])

if __name__ == "__main__":
    # Load data
    labeled_images = pd.read_csv('train.csv')
    images = labeled_images.iloc[0:5000,1:]
    images[images>0]=1  # Pixels with value become 1, otherwise 0
    labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = \
        train_test_split(images, labels, train_size=0.8, test_size=0.2, random_state=0)
    
#    plt.hist(train_images.iloc[i])
#    view_img(2)
    # Train and score
    clf = svm.SVC()
    clf.fit(train_images, train_labels.values.ravel())
    print("Score:", clf.score(test_images,test_labels))
    
    # Predict on test data
#    test_data=pd.read_csv('test.csv')
#    test_data[test_data>0]=1
#    results=clf.predict(test_data[0:5000])
#    print(results)