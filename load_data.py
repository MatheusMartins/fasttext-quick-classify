import chromadb
import fasttext
import csv

myModel = fasttext.load_model('./cc.en.300.bin')

# initiate chromadb client and collection
chroma_client = chromadb.PersistentClient('./localDB/')
collection = chroma_client.get_or_create_collection("ftCollection") 

# calculate and store embeddings for each csv entry
with open('inputData.csv', 'r') as f:
  csv_reader = csv.reader(f)
  for i, row in enumerate(csv_reader):
    if i > 0:
      # parse row columns
      csv_id = row[0]
      sentence = row[1]

      # calculate embeddings
      embeddings = myModel.get_word_vector(sentence).tolist()

      # store in db collection
      collection.add(
				ids=[csv_id],
				embeddings=embeddings,
				documents=sentence,
				metadatas=[{'inputId':csv_id}]
			)