// Summary variables
var length_ratio = 0.5;
var summary_type = 'highlight'; 

// Current tab ID of Gmail
var tab_id;

/**
 * Possible parameters for request:
 *  action: "xhttp" for a cross-origin HTTP request
 *  method: Default "GET"
 *  url   : required, but not validated
 *  data  : data to send in a POST request
 *
 * The callback function is called upon completion of the request */
chrome.runtime.onMessage.addListener(function(request, sender, callback) {
    // Sends a request to the uMail server for content.js
    // This is needed because of cross-origin and HTTP/S security policies
    // From: http://stackoverflow.com/questions/7699615/cross-domain-xmlhttprequest-using-background-pages/7699773#7699773
    if (request.action == "xhttp") {
        tab_id = sender.tab.id;
        console.log(tab_id);

        var xhttp = new XMLHttpRequest();
        var method = request.method ? request.method.toUpperCase() : 'GET';

        xhttp.onload = function() {
            callback(xhttp.responseText);
        };
        xhttp.onerror = function() {
            // Do whatever you want on error. Don't forget to invoke the
            // callback to clean up the communication port.
            callback();
        };
        xhttp.open(method, request.url, true);
        if (method == 'POST') {
            xhttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        }
        xhttp.send(request.data);
        return true; // prevents the callback from being called too early on return

    // Responds to request from content.js for summary variables
    } else if (request.action == "request_summary_variables") {
        tab_id = sender.tab.id;
        console.log(tab_id);
        callback({'length_ratio': length_ratio, 'summary_type': summary_type});

    // Updates summary variables from popup.js
    } else if (request.action == "update_summary_variables") {
        length_ratio = request['length_ratio'];
        summary_type = request['summary_type'];
        console.log("updated summary variables");
    }
});