N_FOLDS = 5

OHE_COLUMNS = [
               'cat1',
               'cat3', 
            #    'cat4', 
            #    'cat5',
               'cat8',
               ]

ORD_COLUMNS = [
                # 'cat0', 
               'cat1', 
            #    'cat2', 
            #    'cat3',
               'cat5',
               'cat8'] 

FREQ_COLUMNS = ['cat0',
                'cat1',
                'cat2',
                'cat3',
                'cat4',
                'cat5',
                'cat6',
                'cat7',
                'cat8',
                'cat9']
                
SQ_COLUMNS = ['cont0',
              'cont1',
              'cont2',
              'cont3',
              'cont4',
              'cont5',
              'cont6',
              'cont7',
              'cont8',
              'cont9',
              'cont10',
              'cont11',
              'cont12',
              'cont13']

UNUSEFUL_COLUMNS = ['id', 
                    'index', 
                    'target', 
                    'kfold']
