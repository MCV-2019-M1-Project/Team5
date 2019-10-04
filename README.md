# Team5 
Members: Joan Fontanals, Yixiong Yang
Date: 4 Oct 2019

This code is to search images from a large image database (DB) for finding paintings 
in a museum image collection.

Main steps:
- Generate descriptors for all DB images;
- Extract features from query image
- Compute similarities between descriptors from the query image and each DB image descriptor,
and order the DB images according these similarities.

Task 1: Create Museum and query image descriptors (BBDD & QS1)
Task 2: Implement / compute similarity measures to compare images
Task 3: Implement retrieval system (retrieve top K results)
Task 4: Evaluation using map@k
Task 5: Background removal using color (QS2). Compute descriptor on foreground (painting)
Task 6: Evaluation of picture masks and retrieval system (QS2)

###########The structure of the code
©À©¤©¤ Readme.md                   // help
©À©¤©¤ Application                 // main part to solve the result
©¦   ©À©¤©¤ application.py          // run the _main_ to output all the results
©À©¤©¤ data                        // data should be here with the default setting. if not, modify the definitions.py
©À©¤©¤ Evaluation                  // functions for evaluating the result (for Task 4,6)
©¦   ©À©¤©¤ MaskEvaluation.py       // evaluate the mask (for Task 6)
©¦   ©À©¤©¤ RankingEvaluation.py    // evaluate the retrieval system (for Task 4)
©À©¤©¤ ImageDescriptors            // Compute image descriptors and Similarity
©¦   ©À©¤©¤ Histogram.py            // define the histogram of both origin and masked pictures (for Task 1)
©¦   ©À©¤©¤ Similarity.py           // calculate the similarity with 4 different methods (for Task 2)
©À©¤©¤ ImageRetrieval              // rank the most similar pictures
©¦   ©À©¤©¤ Ranking.py              // rank the similarity with different methods (for Task 3)
©À©¤©¤ Mask                        // still improving the mask
©À©¤©¤ definitions.py              // define the path of input, output and some global parameters
©À©¤©¤ qsd1_results.pkl            // Best result for QSD1
©À©¤©¤ qsd2_results.pkl            // Best result for QSD2

