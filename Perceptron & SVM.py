#libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.datasets.samples_generator import make_circles
from sklearn import svm
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import random




#this func reduce the data frame according to the classes we want to examine
def reduce_data_frame(df,Class_col_name,Positive_Class, Negative_Class):
    df_new = df
    rows = df_new.shape[0]
    Y_vec = df_new[Class_col_name]
    counter_positive = 0
    counter_negative = 0
    for i in range(0,rows):
        if Y_vec[i] == Positive_Class:
            counter_positive+=1
        elif Y_vec[i] == Negative_Class:
            counter_negative+=1
            df_new[Class_col_name][i] = -1 #change the negative class to (-1)
        else:
            df_new.drop(index = i, axis = 0, inplace = True)            
    return df_new,counter_positive,counter_negative                 

#real data:
def get_data():
    df_name = str(input("DataBase name: ")); #iris.xlsx
    df = pd.read_excel(df_name)
    features_name = []
    num_of_features = int(input("How many features? "))
    class_coumn_name = str(input("What is the name of the class column? "))
    Positive_Class = int(input("What is the number of the positive class? "))
    Negative_Class = int(input("What is the number of the negative class? "))
    df_new,counter_positive,counter_negative = reduce_data_frame(df,class_coumn_name,Positive_Class, Negative_Class)
    X = np.zeros((len(df_new),num_of_features))
    Y = np.zeros(len(df_new))
    Y = df_new[class_coumn_name]
    for i  in range(0,num_of_features):
        parameter_name = str(input("Featrue name: "))
        features_name.append(parameter_name)
    #SepalWidthCm , SepalLengthCm , PetalWidthCm , PetalLengthCm 
    print("")
    
    X = np.array(X)
    Y = np.array(Y)
    
    for i in range(0,num_of_features):
        X[:,i] = df_new[features_name[i]] #get the X vector from data
    
    return X,Y,num_of_features,features_name,Positive_Class,Negative_Class

def plot_X_Y(X,Y):
    length_of_Data = len(Y);
    for i in range(0,length_of_Data):
        x1 = X[i][0];
        x2 = X[i][1];
        if Y[i] == -1:
            plt.plot(x1,x2,'g*');
        elif Y[i] == 1:
            plt.plot(x1,x2,'r.');
        #else:
        #    plt.plot(x1,x2,'b.');
    plt.xlabel("X1");
    plt.ylabel("X2");
    
 
    
 
#perceptron part:    
def plot_HyperPlane_boundary(W,bias,num_features,X,Y):
    if num_features==2:  
            min_x = X[:,0].min();
            max_x = X[:,0].max(); 
            x_plot = np.array([min_x,max_x]); 
            y_plot = -bias/W[1] + -W[0]/W[1]*x_plot; 
            plt.plot(x_plot,y_plot); #plot the boundary decision line
            plt.title("Perceptron algorithm")
            a = -W[0]/W[1];
            b = -bias/W[1];
            if b>0:
                print("The boundary decision line is: y=%.2fx+%.2f " %(a,b));
            else:
                print("The boundary decision line is: y=%.2fx%.2f " %(a,b));

    print("The weight vector is:",W);
    print("The bias balue is: ", bias);
    print("");


def Perceptron_algo(X,Y,num_features,Class0,Class1,W,bias):
    #initial W and bias: W = zero vec, bias = 0
    error_counter = 0
    err_vec = np.zeros(Y.shape[0])
    for i in range(0,Y.shape[0]):
        dot_product = W@X[i]+bias;  #dot product (w^T)*X , w = (bias w1 w2 ....) X = (1 x1 x2 ....)
        if Y[i]*dot_product<=0: #mistake while Yi*(W@X[i]+bias)<=0
            W = W + Y[i]*X[i];
            bias+=1; 
            error_counter+=1 #update while there is an error
        err_vec[i] = error_counter
    return W,bias,error_counter,err_vec;

def convergence_pereptron(X,Y,num_features,Class0,Class1,W,bias):
    bias = 0
    errors_amount = 1;
    Total_err_vec = np.zeros(1) 
    while errors_amount > 0:
        W,bias,errors_amount,err_vec = Perceptron_algo(X,Y,num_features,Class0,Class1,W,bias);
        Total_err_vec = np.concatenate((Total_err_vec,err_vec))
    plot_HyperPlane_boundary(W,bias,num_features,X,Y); #plot the hyperplane boundary
    plt.show()
    plt.plot(Total_err_vec)
    plt.title("Graph of accumulated errors")
    plt.xlabel("Iteration");
    plt.ylabel("Accumulated Errors");
    plt.show();
    return W,bias



#SVM part:
def plot_HyperPlane_boundary_SVM(W,bias,num_features,X,Y):
    if num_features==2:  
            min_x = X[:,0].min();
            max_x = X[:,0].max(); 
            x_plot = np.array([min_x,max_x]); 
            y_plot = -bias/W[1] + -W[0]/W[1]*x_plot; 
            plt.plot(x_plot,y_plot); #plot the boundary decision line
            
            #upper bound:
            x_plot = np.array([min_x,max_x]); 
            y_plot = (-bias-1)/W[1] - W[0]/W[1]*x_plot; 
            plt.plot(x_plot,y_plot); #plot the boundary decision line
            
            #lower bound:
            x_plot = np.array([min_x,max_x]); 
            y_plot = (-bias+1)/W[1] - W[0]/W[1]*x_plot; 
            plt.plot(x_plot,y_plot); #plot the boundary decision line
            plt.title("SVM algorithm")
            a = -W[0]/W[1];
            b = -bias/W[1];
            if b>0:
                print("The boundary decision line is: y=%.2fx+%.2f " %(a,b));
            else:
                print("The boundary decision line is: y=%.2fx%.2f " %(a,b));

    print("The weight vector is:",W);
    print("The bias balue is: ", bias);
    print("");

def SVM_algo_GD(X,Y,num_features,Class0,Class1,W,bias):
    #initial W and bias: W = zero vec, bias = 0
    C = 15
    iterations = 3000
    lr = 1/5/Y.shape[0] #learning rate
    error_counter = 0
    err_vec = np.zeros(iterations)
    epochs = np.zeros(iterations)
    objective_function = np.zeros(iterations)
    obj = 0
    for j in range(0,iterations):
        sigma_W = np.zeros(num_features);
        sigma_b = 0;
        #compute the hinge function dervative for misclassified samples:
        for i in range(0,Y.shape[0]):     
            dot_product = W@X[i]+bias;  #dot product (w^T)*X , w = (bias w1 w2 ....) X = (1 x1 x2 ....)
            if Y[i]*dot_product<=1: #max(0,1-y_i(W@X_i+bias))=1-y_i(W@X_i+bias)
                sigma_W = sigma_W + (-Y[i]*X[i])
                sigma_b = sigma_b + (-Y[i])
                error_counter += 1
                obj += 1-Y[i]*(W@X[i]+bias)
        objective_function[j] = 0.5*np.dot(W,W) + C*obj
        epochs[j] = j
        err_vec[j] = (Y.shape[0]-error_counter)/Y.shape[0]
        dJ_dW = (W +C*sigma_W)
        dJ_db = C*sigma_b
        W = W - lr*dJ_dW
        bias = bias - lr*dJ_db
        error_counter = 0
        obj = 0
    return W,bias,error_counter,epochs,objective_function,err_vec


def convergence_SVM(X,Y,num_features,Class0,Class1,W,bias):
    bias = 0.3
    errors_amount = 1
    t=1
    while t < 2:
        W,bias,errors_amoun,epochs,objective_function,err_vec = SVM_algo_GD(X,Y,num_features,Class0,Class1,W,bias);
        t += 1
    plot_HyperPlane_boundary_SVM(W,bias,num_features,X,Y); #plot the hyperplane boundary
    plt.show();
    plt.plot(epochs,objective_function)
    plt.title("Objective Function")
    plt.xlabel("Iteration");
    plt.ylabel("Objective Function");
    plt.show();
    plt.plot(epochs,err_vec)
    plt.title("Accuracy Rate")
    plt.xlabel("Iteration");
    plt.ylabel("Accuracy Rate");
    plt.show();
    return W,bias

def decision_rule(W,bias,X,Y):
     dot_product = W@X+bias;
     if dot_product>=0:
         predicted_val = 1
     else:
         predicted_val = -1
     return predicted_val

def TestTheModel(W,bias,X,Y):
    confusion_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(Y.shape[0]):
        confusion_matrix[2][2]+=1
        predicted_val = decision_rule(W,bias,X[i],Y)    
        if predicted_val == 1 and Y[i] == 1:
            confusion_matrix[0][0]+=1  #a[pos][pos]+=1
            confusion_matrix[0][2]+=1  #for predicted val
            confusion_matrix[2][0]+=1  #for real val
        elif predicted_val == 1 and Y[i] == -1:
            confusion_matrix[0][1]+=1  #a[pos][neg]+=1
            confusion_matrix[0][2]+=1  #for predicted val
            confusion_matrix[2][1]+=1  #for real val
        elif predicted_val == -1 and Y[i] == 1:
            confusion_matrix[1][0]+=1  #a[neg][pos]+=1
            confusion_matrix[1][2]+=1  #for predicted val
            confusion_matrix[2][0]+=1  #for real val                
        elif predicted_val == -1 and Y[i] == -1:
            confusion_matrix[1][1]+=1  #a[neg][neg]+=1
            confusion_matrix[1][2]+=1  #for predicted val
            confusion_matrix[2][1]+=1  #for real val  
    success_rate = (confusion_matrix[0][0]+confusion_matrix[1][1])/confusion_matrix[2][2]           
    return confusion_matrix, success_rate
  
def circle_map(X,Y):
    x3 = np.zeros((X.shape[0],1))
    X = np.append(X,x3,axis = 1)
    for i in range(0,X.shape[0]):
        X[i][2] = ((X[i][0])**2+(X[i][1])**2)**0.5 #x3 = (x1^2+x2^2)^0.5 -> radius of the circle
    X = np.delete(X,1,axis = 1)
    return X,Y


#text classification part:
def preprocessing(df):    
    # 1 - Remove blank rows if any:
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)
    
    # 2 - Change all the text to lower case:
    df['text'] = [entry.lower() for entry in df['text']]
    
    # 3 - Tokenization : turn the text to words (string) list:
    df['text']= [word_tokenize(entry) for entry in df['text']]
    
    # 4 - Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting:
        
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. 
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV    
    
    for index,entry in enumerate(df['text']):
    #Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and if it include only alphabet letters
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        df.loc[index,'text_final'] = str(Final_words)
    return df 
    
    
def TextClassification():
    #get data:
    df_origin = pd.read_excel("racismdetection.xlsx")
    
    #preprocessing:
    df = preprocessing(df_origin)
    
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['Label'],test_size=0.3)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(df['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    #convert from sparse matrix to numpy arrays:    # Convert sparse TF-IDF matrices to dense arrays
    Train_X_Dense = Train_X_Tfidf.toarray()
    Test_X_Dense = Test_X_Tfidf.toarray()
    
    """
    #print(Tfidf_vect.vocabulary_)
    print("Hi")
    print("TF-IDF matrix:", Train_X_Dense)    
    """
    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=15, kernel='linear', degree=1, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy by python function: ",accuracy_score(predictions_SVM, Test_Y)*100,"%")


    Train_Y[Train_Y == 0] = -1
    Test_Y[Test_Y == 0] = -1
    

    #My SVM:
    W = np.random.rand(Train_X_Dense.shape[1])
    W,bias = convergence_SVM(Train_X_Dense,Train_Y,Train_X_Dense.shape[1],-1,1,W,0)
    
    confusion_matrix, success_rate = TestTheModel(W,bias,Test_X_Dense,Test_Y)
    print("the confusion matrix is: ")
    print(confusion_matrix)
    print("the accuracy rate is: ")
    print("%.3f" %(success_rate*100) + "%")
    print()




        
def main():
    """the answers:
    iris.xlsx
    2
    Class
    1
    0
    choose two of: SepalWidthCm , SepalLengthCm , PetalWidthCm , PetalLengthCm 
    """

    #part 1 - Perceptron Algorithm:
    print("Perceptron's Results:")
    #run for the iris data from excel:
    X,Y,num_features,features_name,Positive_Class,Negative_Class = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=33) 
    plot_X_Y(X_train,y_train)
    W = np.random.rand(num_features)
    bias = 0
    W, bias = convergence_pereptron(X_train,y_train,num_features,Negative_Class,Positive_Class,W,bias)
    confusion_matrix,success_rate = TestTheModel(W,bias,X_test,y_test)
    print("the confusion matrix is: ")
    print(confusion_matrix)
    print("the accuracy rate is: ")
    print("%.3f" %(success_rate*100) + "%")
    print()
    
    #part 2 - SVM Algorithm:
    print("SVM's Circle Results:")  
    #circle:
    plt.show();
    #generate random points - circle sepeartion
    X1,Y1 = make_circles(n_samples=(500,500), random_state=3, noise=0.09, factor = 0.5)       
    Y1[Y1==0] = -1 #change the negative class to -1
    plt.show();
    plot_X_Y(X1,Y1)
    plt.show();
    X,Y = circle_map(X1,Y1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=33)
    plot_X_Y(X_train,y_train)
    W = np.zeros(2)
    bias = 0
    W, bias = convergence_SVM(X_train,y_train,2,-1,1,W,bias)
    confusion_matrix,success_rate = TestTheModel(W,bias,X_test,y_test)
    print("the confusion matrix is: ")
    print(confusion_matrix)
    print("the accuracy rate is: ")
    print("%.3f" %(success_rate*100) + "%")
    print()

    #part 2 - Text Classification Using SVM:
    TextClassification()
    
main()