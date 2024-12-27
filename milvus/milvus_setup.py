from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connetti a Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Definisci schema della collezione
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Usa dimensione embedding appropriata
    FieldSchema(name="metadata", dtype=DataType.JSON)  # Per memorizzare dati aggiuntivi
]
schema = CollectionSchema(fields, description="Collezione di embedding e-commerce")

# Crea la collezione
collection = Collection("ecommerce_embeddings", schema=schema)
collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
print("Collezione creata con successo.")
