#Hannah Daniel, hd892
#CS-UY 4613: Artificial Intelligence Project #2
#12/24/18

#Import numpy library
import numpy as np
learning_rate = 0.1 #learning rate
num_hiddenlayers = 50  #number of hidden layers
MSE = [] #array to hold mean squared error(MSE) over all images, to be used to calculate average
bias = np.ones(1) #bias constant added to sum of weights

#Sigmoid activation function
def sigmoid(num):
    return 1/(1+np.exp(-num)) #1/(1+e^(-num))

#Calculate the average mean squared error over the total number of images in the training set
def calculateAvgMSE():
    avg_MSE = 0
    for value in MSE:
        avg_MSE += value
    avg_MSE = avg_MSE/(len(MSE)) #average of the MSE sums
    MSE.clear() #clears MSE array for next run/iteration
    return avg_MSE

#Calculate back propagation between hidden and input layers
def backProgHidden(weight_hidden, weight_output, delta, hidden_values, inputs):
    #compute delta of hidden layer which is the product of sum of deltas (Wji * delta(i)) and g'(inj)
    delta_hidden = []
    sum_delta = 0
    #stores values of sums of (output weight(Wji) * output value(delta(i)))
    deltas = []
    for j in range(num_hiddenlayers):
        for i in range(5):
            #sum of (output weight(Wji) * output value(delta(i))) for each i
            sum_delta += weight_output[i][j]*delta[i]
        deltas.append(sum_delta)
        sum_delta = 0 #reset value
    #g'(inj) = g(inj)(1-g(inj)) = hidden_values[i]*(1-hidden_values[i])
    for i in range(len(hidden_values)):
        delta_hidden.append(deltas[i]*hidden_values[i]*(1-hidden_values[i]))
    #update weights with formula: Wkj = Wkj + akj*deltaj*learning rate
    for j in range(num_hiddenlayers):
        for k in range(len(inputs)):
            weight_hidden[j][k] += learning_rate*delta_hidden[j]*inputs[k]
    #return updated input to hidden layer weights
    return weight_hidden

#Calculate back propagation between output and hidden layers
def backProgOutput(labels, weight_output, output_values, hidden_values):
    error = [] #store errors of each output value
    mean_sq = 0 #compute MSE/squared error
    for i in range(5):
        #calculate error between predicted and actual
        error.append(labels[i]-output_values[i])
        mean_sq += error[i]**2
    #MSE or squared error = 1/2(sum of error squared), add to array MSE for each image
    MSE.append(mean_sq/2)
    delta = []
    for i in range(5):
        #compute delta as the product of error and sigmoid derivative (output_values[i]*(1-output_values[i]))
        delta.append(error[i]*output_values[i]*(1-output_values[i]))
    #update weights with formula: Wji = Wji + aji*deltai*learning rate
    for i in range(5):
        for j in range(num_hiddenlayers):
            weight_output[i][j] += learning_rate*delta[i]*hidden_values[j]
    #return updated hidden to output layer weights and deltas to update hidden layer weight
    return (weight_output, delta)

#Conduct feed forward propagation
def feedForward(inputs, weight_hidden, weight_output):
    output_values = [] #store output layer values(ai)
    hidden_values = [] #store hidden layer values(aj)
    sum_output = 0
    sum_hidden = 0
    for j in range(num_hiddenlayers):
        for k in range(len(inputs)):
            #compute sum of weights*input values
            sum_hidden += weight_hidden[j][k]*inputs[k]
        #computes a sigmoid of the sum as well as bias and stores it
        hidden_values.append(sigmoid(sum_hidden+bias[0]))
        sum_hidden = 0 #reset value for each hidden layer value
    for i in range(5):
        for j in range(num_hiddenlayers):
            #compute sum of weights*hidden values and takes sigmoid of sum
            sum_output += hidden_values[j]*weight_output[i][j]
        output_values.append(sigmoid(sum_output))
        sum_output = 0 #reset value for each output layer value
    return(hidden_values, output_values, weight_hidden, weight_output)

#Training the network over varying number of epochs until mean squared error difference is < 0.001
def training(list_images, list_labels):
    #these variables keep track of number of epochs, MSE difference as well as curreny and old MSE to compute difference
    epoch = 0
    mse_diff = 0
    old_mse = 0
    current_mse = 0
    #indices to compute data in portions, calculates error for 2561 images or less at a time
    index1 = 0
    index2 = 2561
    #first epoch which does not have hidden layer nor output layer eights
    if epoch == 0:
        weight_hidden, weight_output = trainingSet(list_images,list_labels,index1,index2)
        #calculate average MSE over all images
        old_mse = calculateAvgMSE()
        #print results to console
        print (str(epoch)+ "(img:" + str(index1)+"-" + str(index2)+ ")" + "      " + str(old_mse)+ "         NONE")
        #increases epoch and indices values
        epoch += 1
        index1 = index2
        index2 += index2
    while True:
        #checks to see if index2 goes out of bounds(beyond the total # of images)
        if(len(list_images)-index1) <= 2561 or index2 > len(list_images):
            index2 = len(list_images)
        #trains network for given images
        weight_hidden, weight_output = trainingSet(list_images,list_labels, index1, index2, weight_hidden, weight_output)
        current_mse = calculateAvgMSE()
        #calculates absolute value of MSE difference
        mse_diff = abs(current_mse - old_mse)
        #printing out the result of the each epoch with mean squared error and difference with the previous run's MSE
        print (str(epoch)+ "(img:" + str(index1)+"-" + str(index2)+ ")" + "        " + str(current_mse)+'   ' + str(mse_diff))
        #checks if the difference in MSEs barely changes (threshold: 0.001) or mse < 0.01
        #and the network has been trained with all training set images once, then stop training
        if (mse_diff <0.001 or current_mse < 0.01) and epoch > (len(list_images)/2561):
            print ("Complete!")
            return (weight_hidden, weight_output) #return final hidden and output layer weights for testing, if true
        #reset after going through training set once so training begins with the first portion of images
        if index2 == len(list_images):
            index1 = 0
            index2 = 2561
        #else update indices to move to next portion of training images
        else:
            index1 = index2
            index2 += 2561
        #increment epoch number and set current MSE to old MSE for next iteration
        old_mse = current_mse
        epoch+=1
        

#Training the images through forward and backward propagation
def trainingSet(list_images, list_labels, index1,index2, weight_hidden = None, weight_output = None):
    #sets hidden and output layer weights if non-existent(use numpy.random.rand)
    if weight_hidden == None:
        weight_hidden = [[np.random.randn() for j in range(len(list_images[0]))] for i in range(num_hiddenlayers)]
    if weight_output == None:
        weight_output = [[np.random.randn() for j in range(num_hiddenlayers)] for i in range(5)]
    #conduct feed forward and back propgation for given set of images
    for i in range(index1,index2):
        hidden, output, weight_hidden, weight_output = feedForward(list_images[i], weight_hidden, weight_output)
        weight_output, delta = backProgOutput(list_labels[i], weight_output, output, hidden)
        weight_hidden = backProgHidden(weight_hidden, weight_output, delta, hidden, list_images[i])
    #return resulting hidden and output layer weights
    return (weight_hidden, weight_output)

#Returns index of the max output value and index of the label to index into and increment the confusion matrix
def findMax(labels, output):
    max_output = output[0]
    index_predicted = 0 #saves index of max output values
    index_actual = 0 #saves index of label
    #finds positions of 1 in label(the digit) and max output value 
    for i in range(len(output)):
        if labels[i] == 1:
            index_actual = i
        if max_output < output[i]:
            max_output = output[i]
            index_predicted = i
    return (index_actual, index_predicted)

#Places images and pixels into arrays
def imageData(images):
    list_images = [] #stores array of pixels for each image
    list_pixels = [] #stores pixels for each image
    count = 0 #records number of times for looping
    #iterate through each pixel and add to list
    for pixel in images:
        list_pixels.append(pixel/255)
        count += 1
        #checks if all the pixels of a single image have been added 
        if count == 784:
            count = 0
            list_images.append(list_pixels)
            list_pixels = []
    return list_images

#Places labels into array
def labelData(labels):
    list_labels = []
    #iterate through each label and add to list
    for line in labels:
        label = []
        #convert string to ints for labels for easier computation
        for output in line.strip().split(" "):
            label.append(int(output))
        list_labels.append(label)
    return list_labels


def main():
    #open training image and label files and parse into arrays
    images = open("train_images.raw","rb")
    labels = open("train_labels.txt","r")
    pixels = images.read()
    list_labels = labelData(labels)
    #close files
    images.close()
    labels.close()
    list_images = imageData(pixels)
    #training of the network with training data, trained in epochs using mean-squared error(MSE)
    print ("Epoch/Image Number          MSE                MSE Difference")
    #retrieve final weights for network between input,hidden and output layers
    weight_hiddenfinal, weight_outputfinal = training(list_images,list_labels)
    #open training image and label files and parse into arrays
    images = open("test_images.raw","rb")
    labels = open("test_labels.txt","r")
    pixels = images.read()
    list_labelstest = labelData(labels)
    images.close()
    labels.close()
    list_imagestest = imageData(pixels)
    #create the confusion matrix
    print("\nConfusion Matrix:\n")
    confusion_matrix = [[0 for i in range(5)] for i in range(5)]
    #conduct feed forward propagation for all test images and compute confusion matrix by finding what the network predicted
    for i in range(len(list_imagestest)):
        hidden, output, weight_hidden, weight_output = feedForward(list_imagestest[i],weight_hiddenfinal,weight_outputfinal)
        index_actual, index_predicted = findMax(list_labelstest[i], output)
        #increment for each prediction
        confusion_matrix[index_actual][index_predicted] += 1
    for row in confusion_matrix:
        print(row)
    #compute the accuracy from the confusion matrix, sum of values in diagonal of confusion matrix/total number of images
    accuracy = 0
    #sum of values in diagonal of confusion matrix
    for i in range(5):
        accuracy += confusion_matrix[i][i]
    accuracy = (accuracy/len(list_imagestest)) * 100
    #print accuracy computed
    print("Accuracy: " + str(accuracy) + "%")
    
#Runs main function
if __name__ == "__main__":
    main()

