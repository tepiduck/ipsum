# Bundled text inputs

Text-model preparation reads the bundled public-domain text of *Alice's Adventures
in Wonderland* (Lewis Carroll, 1865). It does not contact Project Gutenberg at runtime.
Model checkpoints, configuration and tokenizer assets may still need to be downloaded.

The source is [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11),
from the copy retained during the 0.11 validation. The bundled file contains the novel
from its opening paragraph through its ending, excluding the ebook header and footer.
Its UTF-8 SHA-256 is
`ee916c3a0ce79fc528621480fd5123f9b55258f0e5938a6be6a04a0defeac06d`.
Preparation verifies that checksum and records it in the input source manifest.
The text is public domain; Faultline's MIT license applies to its code.

The first two paragraphs and the existing disjoint 200-word passage selection are
unchanged. Text batches share a book and are not independent samples of general language
use. Larger batch requests fail if there are insufficient source words. Existing frozen
bundles are replayed from their own inputs and retain their original hashes and manifests.
