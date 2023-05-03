import {parse} from "csv-parse"
import {createReadStream} from "fs"
import {Subject} from "rxjs"

export const loadCSV = <TRow extends {}>(file: string) => {
	const subject = new Subject<TRow>()
	createReadStream(file)
		.pipe(parse({columns: true, bom: true, ignore_last_delimiters: true, quote: ""}))
		.on("data", function (row: TRow) {
			subject.next(row)
		})
		.on("end", function () {
			// acc(rows)
			subject.complete()
		})
		.on("error", function (error: any) {
			console.log(error.message)
			subject.error(error)
		})
	return subject
}
