export const debounce = <T extends any[]>(func: (...args: T) => any, wait: number, immediate: boolean = false) => {
	let timeout: any = null
	return function (...args: T) {
		// const context = this
		clearTimeout(timeout)
		timeout = setTimeout(function () {
			timeout = null
			if (!immediate) func(...args)
		}, wait)
		if (immediate && !timeout) func(...args)
	}
}
