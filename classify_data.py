import chromadb
import fasttext
import numpy as np

myModel = fasttext.load_model('./cc.en.300.bin')

# initiate chromadb client and collection
chroma_client = chromadb.PersistentClient('./localDB/')
collection = chroma_client.get_or_create_collection("ftCollection") 

def cosineSimilarity(A: list,B: list):
  return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def rankInput(input):
  print('\nRanking data against: "' + input + '"')
  inputEmb = myModel.get_sentence_vector(input)
  allDocs = collection.get() 

  # loop through each existing doc to calculate similarity
  calculatedDocs = []
  for docId in allDocs['ids']:
    getDoc = collection.get(ids=[docId], include=['documents', 'embeddings'])
    dataObj = {
      'sentence':  getDoc['documents'][0],
      'similarity': cosineSimilarity(inputEmb, getDoc['embeddings'][0])
    }
    calculatedDocs.append(dataObj)

  # sort by similarity
  sorted_sentences = sorted(calculatedDocs, key=lambda x: x['similarity'], reverse=True)

  # print from most similar to least similar
  for i, obj in enumerate(sorted_sentences):
    print("Rank " + str(i+1) + " - " + obj['sentence'] + ' - ' + str(obj['similarity']))

rankInput('eats carrots')
rankInput('easter')
rankInput('barks at random people')

rankInput('brown')
rankInput('blue')

rankInput("man's best friend")
rankInput("man's worst enemy")