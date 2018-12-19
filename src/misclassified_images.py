"""
Following is the code that can be used to see wrongly-classified images
"""

"""
fnames = test_generator.filenames

ground_truth = test_generator.classes

label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

predictions = model.predict_classes(test_features)
prob = model.predict(test_features)

errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nTest))


print(predictions[errors[0]],ground_truth[errors[0]] )

for i in range(10):#len(errors)):
    pred_class_index = np.argmax(prob[errors[i]]) #predicted class is the one corresponding to which CNN gives the maximum probability
    pred_class_label = idx2label[pred_class_index]
    actual_class_index=ground_truth[errors[i]]
    acutal_class_label=idx2label[actual_class_index]
    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        acutal_class_label,
        pred_class_label,
        prob[errors[i]][pred_class_index]))
    original = load_img('{}/{}'.format("../data/images/",fnames[errors[i]]))
    plt.imshow(original)
    plt.show()

"""
