function plot(plotType) {
    var data = {};
    if (plotType === 'univariate') {
        var column = $('#univariate-column').val();
        data = {
            plot_type: 'univariate',
            column: column
        };
    } else if (plotType === 'bivariate') {
        var column1 = $('#bivariate-column1').val();
        var column2 = $('#bivariate-column2').val();
        data = {
            plot_type: 'bivariate',
            column1: column1,
            column2: column2
        };
    } else if (plotType === 'multivariate') {
        var columns = $('#multivariate-columns').val();
        data = {
            plot_type: 'multivariate',
            columns: columns
        };
    }

    $.ajax({
        type: 'POST',
        url: '/plot',
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: function(response) {
            if (response.hasOwnProperty('html_fig')) {
                var html_fig = response.html_fig;
        
                if (plotType === 'univariate') {
                    $('#univariate-figure-display').html(html_fig);
                } else if (plotType === 'bivariate') {
                    $('#bivariate-figure-display').html(html_fig);
                } else if (plotType === 'multivariate') {
                    $('#multivariate-figure-display').html(html_fig);
                }
            } else {
                console.log('Invalid response from the server');
            }
        },
        error: function(error) {
            console.log(error);
        }
    });
}
