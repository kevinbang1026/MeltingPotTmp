from ret_model import MeltingPotRetriever, MeltingPotEncoder
from pprint import pprint
import argparse
import torch
import os
import torch

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.document import Document

def QA(config):
	OPENAI_API_KEY = config.api_key
	os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

	llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")

	checkpoint_path = config.checkpoint

	checkpoint = torch.load(checkpoint_path, 
							# map_location=torch.device('cpu')
							)
	model = MeltingPotEncoder()
	model.load_state_dict(checkpoint)

	retriever = MeltingPotRetriever(model=model, device=config.device)

	# 사전 제공된 정보들
	query = config.query
	query_emb = retriever.get_embedding(content=query, emb_type="query").tolist()

	# collection이 벡터db
	collection = make_db(retriever)

	out = collection.query(
		query_embeddings=query_emb,
		n_results=10,
	)

	context_doc = 'Content: ' + ("\nContent: ".join((out['documents'][0])))

	doc_list = [Document(page_content=x) for x in out['documents'][0]]

	text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
	docs = [Document(page_content=x) for x in text_splitter.split_text(context_doc)]

	# print(docs)

	template1 = """
	Use the following pieces of context to answer the question at the end.
	If you have something additional information based on the content of the post and it's relevant to the question,
	you can answer it (but you should mention that it's an add-on).

	Also, user can re-question to you with some info related to those documents.
	If you don't know the answer, just say that you don't know, don't try to make up an answer.
	ALWAYS return your answer in Korean.

	Context:
	{context}

	Question:
	{question}

	Answer:
	"""

	template2 = """Given the following extracted parts of a long document and a question, create a final answer with references. 
	If you don't know the answer, just say that you don't know. Don't try to make up an answer.
	ALWAYS return your answer in Korean.

	QUESTION: Which state/country's law governs the interpretation of the contract?
	=========
	Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
	Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
	Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
	=========
	FINAL ANSWER: This Agreement is governed by English law.

	QUESTION: What did the president say about Michael Jackson?
	=========
	Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
	Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
	Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
	Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
	=========
	FINAL ANSWER: The president did not mention Michael Jackson.

	QUESTION: {question}
	=========
	{context}
	=========
	FINAL ANSWER:"""

	# 저장할 텍스트 문자열
	QUESTION_PROMPT1 = PromptTemplate(
		 template= template1, 
		 input_variables= ['context', 'question']
	)

	qa = load_qa_chain(llm=llm, chain_type="stuff", prompt=QUESTION_PROMPT1)

	output = qa({"input_documents": doc_list, "question": query}, return_only_outputs=False)

	print(query)
	print(output['output_text'])

def make_db(retriever):
	# Passage embedding function 정의
	import chromadb
	import json
	from langchain.text_splitter import RecursiveCharacterTextSplitter
	from chromadb import Documents, EmbeddingFunction, Embeddings

	# 매개변수 오류 있어서 texts -> input 으로 변경 : https://github.com/chroma-core/chroma/issues/1388 참고
	class MeltingPotEmbeddingFunction(EmbeddingFunction):
		def __call__(self, input: Documents) -> Embeddings:
			embeddings = retriever.get_embedding(content=input, emb_type='passage')

			return embeddings.tolist()

	default_ef = MeltingPotEmbeddingFunction()

	chroma_client = chromadb.PersistentClient('./chroma')

	collection = chroma_client.get_or_create_collection(
		name='melting_pot',
		metadata={"hnsw:space": "cosine"}, # l2 is the default
		embedding_function=default_ef
	)

	# DB에 내용을 마련하기 위해 추가하는 함수들

	# with open(config.dummy + '/dummy_articles.json', 'r') as file:
	# 	articles = json.load(file)

	# text_splitter = RecursiveCharacterTextSplitter(
	# 	chunk_size = 2000,
	# 	chunk_overlap  = 500,
	# 	length_function = len,
	# )

	# for idx, article  in enumerate(articles):
	# 	# Split the article content into chunks
	# 	chunks = text_splitter.split_text(article['content'])

	# 	# Prepare document and metadata lists for adding to ChromaDB
	# 	documents = []
	# 	embeddings = []
	# 	for ch in chunks:
	# 		raw_text = article["title"] + " - " + ch
	# 		embeddings.append(default_ef(raw_text)[0])
	# 		documents.append(ch)

	# 	metadatas = [{'source': article['title'], "seg" : i} for i in range(len(chunks))]
	# 	ids = [f"{idx}_{i}" for i in range(len(chunks))]

	# 	# Add chunks to the ChromaDB collection
	# 	collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

	# 	print(f"completed adding {idx}")

	# 	if idx > 15:
	# 		break
		
	return collection
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--checkpoint', type=str, default='/content/checkpoints/4__meltingPot-model_optim.pt')
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--dummy', type=str, default='/content/dummy')
	parser.add_argument('--chroma', type=str, default='/content/chroma')
	parser.add_argument('--query', type=str, default='알고리즘 정리된 것들 설명해줘 어떤게 있었지?')
	parser.add_argument('--api_key', type=str, default='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')
	
	
	config = parser.parse_args()
  
	QA(config)