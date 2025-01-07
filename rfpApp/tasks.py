from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery.utils.log import get_task_logger
from rfpApp.engine.document_parser import DocumentParser
logger = get_task_logger(__name__)

@shared_task(bind=True)
def vectorSearch(self, query_text):
    dparser = DocumentParser(document_type='pdf')
    response = dparser.recursive_indexing(query_text)
    return response