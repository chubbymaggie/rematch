import itertools
import json

import sklearn as skl
import sklearn.metrics  # noqa flake8 importing as a different name
import sklearn.preprocessing  # noqa flake8 importing as a different name
import sklearn.feature_extraction  # noqa flake8 importing as a different name

from . import match


class HistogramMatch(match.Match):
  @staticmethod
  def match(source, target):
    source_values = itertools.izip(*source.values_list('id', 'instance_id',
                                                       'data'))
    target_values = itertools.izip(*target.values_list('id', 'instance_id',
                                                       'data'))

    source_ids, source_instance_ids, source_data = source_values
    target_ids, target_instance_ids, target_data = target_values

    dictvect = skl.feature_extraction.DictVectorizer()
    source_data = dictvect.fit_transform([json.loads(d) for d in source_data])
    target_data = dictvect.transform([json.loads(d) for d in target_data])

    source_matrix = skl.preprocessing.normalize(source_data, axis=1, norm='l1')
    target_matrix = skl.preprocessing.normalize(target_data, axis=1, norm='l1')

    distance_matrix = skl.metrics.pairwise.pairwise_distances(source_matrix,
                                                              target_matrix)

    for source_i in range(distance_matrix.shape[0]):
      source_id = source_ids[source_i]
      source_instance_id = source_instance_ids[source_i]

      for target_i in range(distance_matrix.shape[1]):
        target_id = target_ids[target_i]
        target_instance_id = target_instance_ids[target_i]

        distance = distance_matrix[source_i][target_i]
        score = (1 - distance) * 100
        yield (source_id, source_instance_id, target_id, target_instance_id,
               score)
