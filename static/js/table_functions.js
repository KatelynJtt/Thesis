function toggleTable() {
    var table = document.getElementById("dataTable").getElementsByTagName("table")[0];
    if (table) {
        if (table.style.display === "none") {
            table.style.display = "table";
        } else {
            table.style.display = "none";
        }
        document.getElementById("message").innerHTML = "";
    } else {
        document.getElementById("message").innerHTML = "No table to toggle.";
    }
}

function clearTable() {
    var table = document.getElementById("dataTable").getElementsByTagName("table")[0];
    if (table) {
        table.innerHTML = "";
        document.getElementById("message").innerHTML = "";
    } else {
        document.getElementById("message").innerHTML = "No table to clear.";
    }
}

function downloadTable() {
    var table = document.getElementById("dataTable").getElementsByTagName("table")[0];
    if (table) {
        var csv = [];
        var rows = table.getElementsByTagName("tr");

        for (var i = 0; i < rows.length; i++) {
            var row = [], cols = rows[i].querySelectorAll("td, th");
            for (var j = 0; j < cols.length; j++)
                row.push(cols[j].innerText);
            csv.push(row.join(","));
        }

        var csvData = new Blob([csv.join("\n")], { type: 'text/csv' });
        var csvUrl = URL.createObjectURL(csvData);
        var hiddenElement = document.createElement('a');
        hiddenElement.href = csvUrl;
        hiddenElement.target = '_blank';
        hiddenElement.download = 'table.csv';
        hiddenElement.click();
        document.getElementById("message").innerHTML = "";
    } else {
        document.getElementById("message").innerHTML = "No table to download.";
    }
}