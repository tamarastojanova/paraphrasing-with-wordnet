def run_query(query, driver, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record.data() for record in result]


def run_similarity_query_wo_examples(query_embedding, relations, driver):
    query = f"""
    WITH $query_embedding AS query_embedding
    MATCH (synset1:Synset)-[r]->(synset2:Synset)
    WHERE type(r) IN $relations
    AND synset1.embedding IS NOT NULL AND synset2.embedding IS NOT NULL

    WITH DISTINCT synset2,
        COLLECT(DISTINCT type(r)) AS relations,  // Collect all relations for each synset
        reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (query_embedding[i] * synset2.embedding[i])) AS dot_product,
        reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (query_embedding[i] * query_embedding[i])) AS norm_query,
        reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (synset2.embedding[i] * synset2.embedding[i])) AS norm_synset2

    WITH synset2, dot_product / (sqrt(norm_query) * sqrt(norm_synset2)) AS similarity, relations

    RETURN DISTINCT synset2.name AS synset_name,
                    synset2.definition AS synset_definition,
                    similarity,
                    relations  // Now returns a list of relations
    ORDER BY similarity DESC
    LIMIT 5
  """

    parameters = {
        "query_embedding": query_embedding,
        "relations": relations
    }

    results = run_query(query, driver, parameters)
    return results


def run_similarity_query_w_examples(query_embedding, relations, driver):
    query = f"""
      WITH $query_embedding AS query_embedding

      MATCH (synset1:Synset)-[r]->(synset2:Synset)
      WHERE type(r) IN $relations
      AND synset1.embedding IS NOT NULL AND synset2.embedding IS NOT NULL

      WITH DISTINCT synset2,
          type(r) AS relation,
          reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (query_embedding[i] * synset2.embedding[i])) AS dot_product,
          reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (query_embedding[i] * query_embedding[i])) AS norm_query,
          reduce(sum = 0, i IN range(0, size(query_embedding)-1) | sum + (synset2.embedding[i] * synset2.embedding[i])) AS norm_synset2

      WHERE norm_query > 0 AND norm_synset2 > 0

      WITH synset2, dot_product / sqrt(norm_query * norm_synset2) AS similarity, relation
      ORDER BY similarity DESC
      LIMIT 10

      OPTIONAL MATCH (synset2)<-[:EXAMPLE]-(example:Example)

      WITH synset2, similarity,
           [x IN COLLECT(relation) WHERE x IS NOT NULL] AS relations,
           [x IN COLLECT(example.text) WHERE x IS NOT NULL][..5] AS examples

      RETURN DISTINCT
          synset2.name AS synset_name,
          synset2.definition AS synset_definition,
          similarity,
          relations,
          examples
      ORDER BY similarity DESC
      LIMIT 5
    """

    parameters = {
        "query_embedding": query_embedding,
        "relations": relations
    }

    results = run_query(query, driver, parameters)
    return results
