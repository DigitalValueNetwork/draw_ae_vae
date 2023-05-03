import {BinaryToTextEncoding, createHmac} from "crypto"

/** Wrapper for Object.entries with typed output.  Note: Only correct when you know the object only contains the given properties */
// https://stackoverflow.com/questions/55012174/why-doesnt-object-keys-return-a-keyof-type-in-typescript
export const getObjEntries = <T extends {}>(obj: T) => Object.entries(obj) as [keyof T, any][]

const SALT = "$ome$alt"
export const generateHash = (str: string, length: number, digestEncoding: BinaryToTextEncoding = "base64url") => createHmac("sha256", SALT).update(str).digest(digestEncoding).substring(0, length)


/** Load filename from arguments, failing if it's not found */
export const getFilenameFromLastArgument = (argv: string[], idxFromEnd = 0, optional = false) => {
	const fileName = argv[argv.length - (1 + idxFromEnd)]
	if (optional && (argv.length < 3 + idxFromEnd || fileName.substring(0, 2) === "--"))
		return ""

	if (!optional && ((fileName ?? "") == "" || argv.length < 3 + idxFromEnd)) {
		console.error("No file name provided, or incorrect number", argv, argv.length, fileName)
		process.exit(99)
	}
	return fileName
}
