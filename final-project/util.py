import pickle

class GeneralUtils():

    def __init__(self,config):

        self.config       = config
        self.raw_text     = None
        self.refined_text = None

        self.sentence2class = None
        self.word2idx       = None
        self.idx2word       = None
        self.char2idx       = None
        self.idx2char       = None

        self.char_num  = None
        self.word_num  = None
        self.sentence_num = None

        self.char_hidden_size = None
        self.word_hidden_size = None
        
        if(config.load_dic == True):
            self.LoadData()
        else:
            self.BuildData()

    def LoadData(self):
        with open(self.config.dictionary_path,"rb") as f:
            data_list = []
            while True:
                try:
                    data = pickle.load(f)
                except EOFError:
                    break
                data_list.append(data)

        self.sentence2class = data_list[0][0]
        self.word2idx       = data_list[0][1]
        self.idx2word       = data_list[0][2]
        self.char2idx       = data_list[0][3]
        self.idx2char       = data_list[0][4]

        self.char_num  = len(self.char2idx.values())
        self.word_num  = len(self.word2idx.values())
        self.sentence_num = len(self.sentence2class)

        print(self.char_num)
        print(self.word_num)
        print(self.sentence_num)

    def BuildData(self):

        self.ReadText()
        self.BuildDic()
        self.SaveData()

    def SaveData(self):
        dump_list = [
            self.sentence2class,
            self.word2idx,
            self.idx2word,
            self.char2idx,
            self.idx2char
        ]
        with open(self.config.dictionary_path,"wb") as f:
            pickle.dump(dump_list,f)

        return

    def ReadText(self):
        file = open(self.config.train_data)
        return_value = []
        while(True):
            line = file.readline()
            if(line == ""):
                break
            else:
                return_value.append(line)
        self.raw_text = return_value

    def BuildDic(self):
        sentence2class = []
        word2idx = {}
        idx2word = {}
        char2idx = {}
        idx2char = {}

        for sentence in self.raw_text:
            sen_class  = sentence[0]
            sentence_  = sentence[2:].strip()
            sentence2class.append((sentence_,sen_class))
            #
            for word in sentence_.split():

                if word not in word2idx.keys():
                    word2idx[word] = len(word2idx.keys())
                if word not in idx2word.values():
                    idx2word[len(idx2word)] = word

                for char in word:
                    if char not in char2idx.keys():
                        char2idx[char] = len(char2idx.keys())
                    if char not in idx2char.values():
                        idx2char[len(idx2char)] = char

        self.sentence2class = sentence2class
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.char2idx = char2idx
        self.idx2char = idx2char
