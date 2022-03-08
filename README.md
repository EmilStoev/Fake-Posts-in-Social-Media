Apply different Machine Learning algorithms to determine the best way to differentiate 'fake' from 'real' posts.

To prepare the data I normalised all letters to lowercase, removed the 'stopwords' and the punctuation.

These are my results:

![LSVC](https://user-images.githubusercontent.com/98655631/157280548-51fb4331-b9a5-41e8-9efb-82552d79c765.png)
![DTC](https://user-images.githubusercontent.com/98655631/157280682-b02b0741-4060-421d-99c7-68c012ee9fe5.png)
![RFC](https://user-images.githubusercontent.com/98655631/157280688-2655af3d-a58e-458d-a405-b98b72c49af2.png)
![MNB](https://user-images.githubusercontent.com/98655631/157280689-b7ba2c68-6c75-4832-b5fc-b9549a802d4d.png)
![KNN](https://user-images.githubusercontent.com/98655631/157280690-2af8a06b-72ea-4f7f-af9f-3c0d5c087546.png)
![SGDC](https://user-images.githubusercontent.com/98655631/157280691-d3ec31a0-8a45-46b7-ba6e-c881fd8ace41.png)

These are the Confusion Matrices for all 6 models. The second way to see the total accuracy is to use sklearn's 'accuracy_score', used here:

![Results](https://user-images.githubusercontent.com/98655631/157280894-77f40061-6ff5-4515-911f-4fefc046c1b1.png)

Depending on the needed result, a different model can be described as 'best'. If overall accuracy is needed, then the LinearSVC model proves the best accuracy.
If finding specifically 'fake' posts and not flagging them as 'real' (top right of confusion matrix), then the Random Forest Classifier is the best with only 30 fake posts being flagged as real.
