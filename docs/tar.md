# LINDI binary (tar) format

In addition to a JSON/text format, LINDI offers a binary format packaged as a tar archive, which includes a specialized lindi.json file in the standard JSON format as well as other files including binary chunks. The `lindi.json` file can reference a mix of external references and internal binary chunks.

**General structure of a tar archive**: Tar is a simple and widely-used format that houses binary files sequentially, with each file record beginning with a 512-byte header that describes the file (name, size, etc.), followed by the content rounded up to 512-byte blocks. The archive is terminated by two 512-byte blocks filled with zeros.

**Cloud Optimization**: Tar archives are typically not optimized for cloud storage due to their sequential file arrangement which necessitates reading all headers for index construction. To address this, LINDI introduces two crucial files within each archive:

`.tar_entry.json`: This must always be the first file in the archive, fixed at 1024 bytes (padded with whitespace if necessary). It specifies the byte range for the `.tar_index.json` file, allowing it to be quickly located and read.

`.tar_index.json`: Contains names and byte ranges of all other files in the archive, enabling efficient random access after the initial two requests (one for `.tar_entry.json` and one for `.tar_index.json`).

**Handling Updates and Data Growth**: Traditional tar clients do not allow for file resizing or deletion, posing a challenge when updating files like `lindi.json` that might grow as data is added. LINDI circumvents these issues by padding `lindi.json` and `.tar_index.json` with extra whitespace, allowing for in-place expansion up to a predetermined limit without modifying the tar structure. If expansion beyond this limit is necessary, the original file is renamed to a placeholder (e.g., `./trash/xxxxx`), effectively removing it from use, and a new version of the file is appended to the end of the archive.

**Efficient Cloud Interaction**: With the special structure of `.tar_entry.json` and `.tar_index.json`, clients can download the index with minimal requests, reducing the overhead typical of cloud interactions with large tar archives.