// Place your application-specific JavaScript functions and classes here
// This file is automatically included by javascript_include_tag :defaults
	var mycount;
	var mylabels = new Hash();
	var mysent = new Hash();
	var phase = 0;
	function toggleSent(e)
	{
		id = e.target.id;
		//Toggle Color of sentences
		var toggle = Element.getStyle(id,'backgroundColor');
		if (toggle == 'yellow' || toggle == '#ffff00')
		{
		  $(id).setStyle({backgroundColor: '#ddd'}); 
		  mysent.unset(id);
		} 
		else
		{
		  $(id).setStyle({backgroundColor: 'yellow'}); 
		  mysent.set(id,true);
		}
	}
	function highlight(id){
		$(id).setStyle({backgroundColor: '#ffff99'});
	}
	function lowlight(id){
		$(id).setStyle({backgroundColor: 'white'});
	}
	
	function wordCount(){
		var wordsleft = 251 - document.getElementById('summary_Sum').value.gsub(/\[(\d+\.\d+,)*(\d+\.\d+)\]/,'').split(/\w+/g).length;
		if (wordsleft < 0)
		{
			document.getElementById('counter').innerHTML = "<span style='color:red;'>"+wordsleft+"</span>";
		}
		else
		{
			document.getElementById('counter').innerHTML = wordsleft
		}
		return wordsleft;
	}
	function record_highlights()
	//Counts the selected sentences
	{
		mycount = wordCount();
		if (mycount < 0)
			alert("Please use less words in your summary");
		else if (mycount == 250)
			alert("Please write your own summary");
		else if (!document.getElementById('summary_Sum').value.match(/[.!?)]\s*\[(\d+\.\d+,)*(\d+\.\d+)\]/))
			alert("Please link your written sentences to the original text using the following syntax:\n\
\"I have one summary sentence.[1.2]\"");
		else
		{
			if (phase < 1)
			{
				//Record Written summary
				//Parse and check links
				var sum = document.getElementById('summary_Sum').value
				sum.scan(/\[(\d|\.|,)+\]/,addLink);
				phase = 1;
				alert("Now, please select the sentences from the original emails that should be included in a summary. Your links have been preselected. You can still make changes to your original summary.");
				document.getElementById('i_box').innerHTML = "<b>Summarize: </b>Now, please select the sentences from the original emails that should be included in a summary.";
				document.getElementById('submit').value = "Submit";
				 //onclick='toggleSent(event,\"s#{email_count}.#{count}\");'
				list = document.getElementsByClassName('sentences');
				for (var i=0; i < list.length; i++)
				{
					$(list[i]).onclick = toggleSent;
					if (mysent.get(list[i].id) == true)
						$(list[i]).setStyle({backgroundColor: 'yellow'}); 
				}
			}
			else
			{
				//Encode Selected Sentences
				res = mysent.keys();
				if (res == "")
				{
					alert("You have not selected any sentences as important.")
					return false;
				}
				//set hidden form field
				document.getElementById('summary_Sent').value = res;
				return true;
			}
		}
		return false;
	}
	function addLink(link)
	{
		link[0].scan(/\d+\.\d+/,function(match){mysent.set('s'+match[0],true)});
	}
	function toggleMe(id)
	{
	  //Toggles each of the sentences
	  if (mylabels.get(id) == true)
	  {
	    $(id).src = '/images/'+id.split(/\d/)[0]+'_d.gif';
		mylabels.unset(id);
	  }
	  else
	  {
	    $(id).src = '/images/'+id.split(/\d/)[0]+'_l.gif';
		mylabels.set(id,true);
	  }
	}
	function record_labels()
	{
		if (mylabels.keys().size() == 0)
		{
			alert("You did not select any labels.");
			return false;
		}
		else	
		{
		document.getElementById('summary_Label').value = mylabels.keys();
		if(document.getElementById('title').innerHTML == "Subject: Example")
		{
			//Check the answers
			if (document.getElementById('submit').value != "Continue")
			{
				alert("Now compare your answers with ours to see if you understand the labeling. Our answers will appear lighter.");
				//Set the buttons accordingly
				mylabels = new Hash();
				list = document.getElementsByClassName('req');
				list = list.concat(document.getElementsByClassName('prop'));
				list = list.concat(document.getElementsByClassName('cmt'));
				list = list.concat(document.getElementsByClassName('meet'));
				list = list.concat(document.getElementsByClassName('meta'));
				list = list.concat(document.getElementsByClassName('subj'));
				for (var i=0; i < list.length; i++)
				{
					if (list[i].id == "meet1.2"||
						list[i].id == "meet1.3"||
						list[i].id == "meet1.4"||
						list[i].id == "prop1.4"||
						list[i].id == "meta2.2"||
						list[i].id == "meet2.3"||
						list[i].id == "meet2.4"||
						list[i].id == "meet2.5"||
						list[i].id == "req2.5"||
						list[i].id == "meta2.5"||
						list[i].id == "cmt2.6"||
						list[i].id == "meet2.6"||
						list[i].id == "prop2.7"||
						list[i].id == "subj2.7"||
						list[i].id == "meet3.3"||
						list[i].id == "cmt3.4"||
						list[i].id == "meet4.1"||
						list[i].id == "meet4.2"||
						list[i].id == "cmt4.2"||
						list[i].id == "meta4.4"||
						list[i].id == "subj2.3"||
						list[i].id == "subj2.4"||
						list[i].id == "subj1.3"||
						list[i].id == "subj3.2")
						
					{
						list[i].style.opacity = 0.5;
						mylabels.set(list[i].id,true);
					}
				}
				//Change button label accordingly
				document.getElementById('submit').value = "Continue";
				return false;
			}
			else
				return true;
		}
		return confirm("You are now done with this thread.");
		}
	}
	function change_all(expand)
	{
		//Change from expand to collapse
		Element.toggle('expandbutton');
		Element.toggle('collapsebutton');
		//Change all the emails
		list = document.getElementsByClassName('emailbutton');
		for (var i=0; i < list.length; i++)
		{
			if ((expand && (document.getElementById('e'+list[i].id).style.display == "none")) ||
			   (!expand && (document.getElementById('e'+list[i].id).style.display != "none")))
			{
				Element.toggle('e'+list[i].id);
		    	Element.toggle('open'+list[i].id);
		    	Element.toggle('closed'+list[i].id);
			}
		}
	}