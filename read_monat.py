import pandas as pd

data_df=pd.read_csv('./data/full_data/articles.csv',lineterminator='\n')
data_df.columns=['doc_id', 'title', 'author_id', 'body', 'raw_body', 'category', 'perex', 'url', 'source_id', 'other_info', 'published_at', 'extracted_at', 'feedback_facebook_id', 'monitoring_tags']
labels=pd.read_csv('./data/full_data/relation_annotations.csv')
labels.columns=['id', 'method_id', 'annotation_type_id', 'annotation_category', 'source_entity_type', 'doc_id', 'target_entity_type', 'target_entity_id', 'value', 'created_at', 'updated_at']

data_merged=pd.merge(labels,data_df,on=['doc_id'])
required_columns=['doc_id',
                  'title','body','category','source_id',
                  'annotation_type_id','target_entity_type','target_entity_id','value']

data_merged=data_merged[required_columns]
#
# fact_checking=pd.read_csv("./data/full_data/entity_annotations.csv")
# labels_label=fact_checking[fact_checking['annotation_category']=='label']
#
# sources=pd.read_csv("./data/full_data/sources_types.csv")
# dicussion_post=pd.read_csv("./data/full_data/discussion_posts.csv")
# claims=pd.read_csv("./data/full_data/claims.csv")