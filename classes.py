# Opis klasa za prepoznavanje za dani objekt
# struktura:
#   ime_objekta = [ 
#                   ["ime_klase_1", 
#                     [r'ime_direktorija_za_train_1', r'ime_direktorija_za_train_2', ...], 
#                     [r'ime_direktorija_za_test_1', r'ime_direktorija_za_test_2', ...]
#                   ]
#                   
#                   ["ime_klase_2", 
#                     [r'ime_direktorija_za_train_1', r'ime_direktorija_za_train_2', ...], 
#                     [r'ime_direktorija_za_test_1', r'ime_direktorija_za_test_2', ...]
#                   ]
# #                   
#                   ...
#
#                   ["ime_klase_n", 
#                     [r'ime_direktorija_za_train_1', r'ime_direktorija_za_train_2', ...], 
#                     [r'ime_direktorija_za_test_1', r'ime_direktorija_za_test_2', ...]
#                   ]
# ]

eye_classes = [ 
                ["open", 
                  [
                    r'archive/TrainingData/EyesOnly/Models/Open',
                    r'archive/TrainingData/EyesOnly/Custom/Karlo/Open'
                  ], 
                  [r'archive/TestData/EyesOnly/Custom/Karlo/Open']
                ], 
                ["closed", 
                  [
                    r'archive/TrainingData/EyesOnly/Models/Closed',
                    r'archive/TrainingData/EyesOnly/Custom/Karlo/Closed'
                  ], 
                  [r'archive/TestData/EyesOnly/Custom/Karlo/Closed']
                ]
              ]

mouth_classes = [ 
                ["open", 
                  [
                    r'archive/TrainingData/MouthOnly/Models/Set1/Open',
                    r'archive/TrainingData/MouthOnly/Models/Set2/Open'
                    #r'archive/TrainingData/MouthOnly/Models/Set1_small/Open',
                    #r'archive/TrainingData/MouthOnly/Models/Set2_small/Open'
                  ], 
                  [ 
                    r'archive/TestData/MouthOnly/Custom/Karlo/Open'
                  ]
                ], 
                ["closed", 
                  [
                    r'archive/TrainingData/MouthOnly/Models/Set1/Closed',
                    r'archive/TrainingData/MouthOnly/Models/Set2/Closed'
                    #r'archive/TrainingData/MouthOnly/Models/Set1_small/Closed',
                    #r'archive/TrainingData/MouthOnly/Models/Set2_small/Closed'
                  ], 
                  [
                    r'archive/TestData/MouthOnly/Custom/Karlo/Closed'
                  ]
                ]
              ]