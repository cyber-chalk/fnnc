let json;
let numbersContainer = document.getElementById("numbers");

let darkblue = "rgba(38, 60, 255, 1)";
let lightblue = "rgba(21, 213, 255, 1)";

let coloblur = {
	"#000000": "0px",
	"rgba(38, 60, 255, 1)": "13.4px",
	"rgba(21, 213, 255, 1)": "21px"
};

const format = (container, colour, itr, numberArray) => {
	let formattedHTML = "";
	// let numberArray = numbers.trim().split(/\s+/);
	for (let i = 0; i < 784; i++) {
		const value = numberArray[i].toFixed(1);

		const color = value == 0.0 ? "unset" : colour;
		const className = value == 0.0 ? "zero" : "non-zero";

		const [br, odiv, ediv] =
			i % 28 === 0 ? ["<br>", "<span>", "</span>"] : ["", "", ""];
		const solid = colour == "white" ? "solid" : "";

		formattedHTML += `${ediv} ${odiv}<span style="--bg-color: ${color}; --blur: ${coloblur[colour]} " class="${className} ${solid} number">${value}</span>`;

		// --blur: ${coloblur[colour]}
	}

	container.innerHTML = formattedHTML;
	// if (itr == 0) return;
	container.style.transform = `translate(${-50 + itr * 25}px, ${
		50 - itr * 25
	}px)`;
	// "translate(" + "-5px" + "," + "10px" + ")"; // literals dont work
	// container.style.zIndex = 0 - itr;
	// container.style.opacity = 0.5;
};

function updateText(i) {
	document.getElementById("label").innerText = `Label: ${json.testLabels[i]}`;
	document.getElementById("guess").innerText = `Guess: ${json.rightArr[i]}`;
}

let currentIndex = 0;
const load = (plusminus) => {
	let numberContainer = document.getElementById("numbers");
	if (numberContainer.hasChildNodes())
		[...numberContainer.children].forEach((element) => {
			element.remove();
		});
	for (let i = 0; i < 3; i++) {
		let layer = document.createElement("div");
		layer.classList.add("layer");
		let container = document.getElementById("numbers");
		container.append(layer);
	}
	let container = document.getElementsByClassName(`layer`);
	let colarr = [darkblue, lightblue, "white"];
	// let colarr = ["#000000"];
	currentIndex = (currentIndex + plusminus + 15) % 15;
	let numbers = json.testImages[currentIndex];
	updateText(currentIndex);

	for (let i = 0; i < colarr.length; i++)
		format(container[i], colarr[i], i, numbers);
};
// load(0);

const getJson = async () => {
	let res;
	try {
		res = await fetch("/api");
	} catch (err) {
		console.log(err);
	}
	json = await res.json();
	console.log(json);
	document.getElementById("lossRate").innerText = `Loss Rate: ${json.loss}`;
	load(0);
};
getJson();
