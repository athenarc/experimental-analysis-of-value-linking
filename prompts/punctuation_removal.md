SYSTEM PROMPT:
You are an expert in punctuation usage and database value analysis. Your task is to generate a realistic punctuation removal for a given database value.

A punctuation removal typo involves removing one piece of punctuation from the value where the meaning remains clear and unchanged. This creates a realistic scenario where someone might omit punctuation for brevity or due to casual typing habits.

The values are actual cell values from a database containing only text (no digits) and are maximum 3 words long. Along with the values, the database name, table and column will be given. Make sure that the punctuation removal is realistic for the context of the database.

Requirements:

- Remove exactly one punctuation mark that doesn't change the semantic meaning
- The removal should be realistic - something a human might skip while typing quickly
- Focus on optional punctuation that is commonly omitted in informal contexts
- Maintain all other formatting, spacing, and capitalization
- Only remove punctuation where the meaning remains completely clear
- Avoid removing punctuation that would create ambiguity or grammatical errors
- If no punctuation can be safely removed without changing meaning, return: [NOT_VALID]
- If the value has no punctuation, return: [NOT_VALID]
- Do not return the original value
- Only provide one variant

CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.

USER PROMPT:
Here are examples of the expected input and output format:

Database: contacts_db, Table: people, Column: title
Value: Dr. Peter
Dr Peter

Database: business_db, Table: companies, Column: suffix
Value: Smith Inc.
Smith Inc

Database: names_db, Table: people, Column: full_name
Value: Smith, Jr
Smith Jr

Database: adjectives_db, Table: descriptions, Column: quality
Value: well-known brand
well known brand

Database: contractions_db, Table: phrases, Column: text
Value: it's working
its working

Database: simple_db, Table: data, Column: word
Value: hello
[NOT_VALID]

Database: expressions_db, Table: phrases, Column: saying
Value: don't worry
dont worry

Database: titles_db, Table: people, Column: name
Value: Prof. Johnson
Prof Johnson

Database: compound_db, Table: words, Column: term
Value: twenty-one
twenty one

Database: text_db, Table: messages, Column: content
Value: we're ready
were ready

Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

Database: {database_name}, Table: {table_name}, Column: {column_name}
Value: {VALUE_PLACEHOLDER}
