// Load summary variables from background.js
var length_ratio;
var summary_type;
chrome.runtime.getBackgroundPage(function (backgroundPage) {
    length_ratio = backgroundPage.length_ratio;
    summary_type = backgroundPage.summary_type;
    tab_id = backgroundPage.tab_id;
})

// Listeners for change of summary variables in settings 
window.onload = function () {
    document.getElementById("length-ratio").oninput = function () {
        length_ratio = parseInt(document.getElementById("length-ratio").value) / 100;
        updateSummaryVariables();
    }
    document.getElementById("show-summary").onclick = function () {
        summary_type = 'show-summary';
        updateSummaryVariables();
    }
    document.getElementById("highlight").onclick = function () {
        summary_type = 'highlight';
        updateSummaryVariables();
    }

    // Set settings to display current summary variables
    document.getElementById('length-ratio').value = length_ratio * 100;
    if (summary_type == 'highlight') { document.getElementById('highlight').checked = true; }
    else { document.getElementById('show-summary').checked = true; }
}

// Function to push new summary variables to umail.js
var updateSummaryVariables = function () {
    // Update background.js
    chrome.runtime.sendMessage({
        action: 'update_summary_variables',
        'length_ratio': length_ratio,
        'summary_type': summary_type
    });

    // Update umail.js
    chrome.runtime.getBackgroundPage(function (backgroundPage) {
        var tab_id = backgroundPage.tab_id;
        chrome.tabs.sendMessage(tab_id, {
            action: 'update_summary_variables',
            'length_ratio': length_ratio,
            'summary_type': summary_type
        });
    });   
}

// Make the links in the extension pop up work correctly
document.addEventListener('DOMContentLoaded', function () {
    var links = document.getElementsByTagName("a");
    for (var i = 0; i < links.length; i++) {
        (function () {
            var ln = links[i];
            var location = ln.href;
            ln.onclick = function () {
                if (location != '') {
                    chrome.tabs.create({active: true, url: location});
                }
            };
        })();
    }
});