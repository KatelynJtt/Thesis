function plot(plot_type) {
    var column1, column2;
    if (plot_type === 'histogram' || plot_type === 'box') {
        column1 = $("#" + plot_type + "-column").val();
    } else if (plot_type === 'heatmap') {
        column1 = 'ALL';
    } else {
        column1 = $("#" + plot_type + "-column1").val();
        column2 = $("#" + plot_type + "-column2").val();
    }
    var data = {
        'column1': column1,
        'plot_type': plot_type
    };
    if (column2) {
        data['column2'] = column2;
    }
    $.ajax({
        type: "POST",
        url: "/plot",
        data: JSON.stringify(data),
        contentType: "application/json",
        success: function(response) {
            var graphJSON = response.graphJSON;
            var graphData = JSON.parse(graphJSON);
            Plotly.newPlot(plot_type + '-figure-display', graphData);                
        },
        error: function(err) {
            console.log(err);
        }
    });
}
