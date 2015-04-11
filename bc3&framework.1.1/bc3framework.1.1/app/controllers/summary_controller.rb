class SummaryController < ApplicationController

  # GETs should be safe (see http://www.w3.org/2001/tag/doc/whenToUseGet.html)
  verify :redirect_to => { :action => :list }
  
  #Display email thread for annotations
  def show
    if params[:email_thread]
      @email_thread = EmailThread.find(params[:email_thread])
    elsif session[:id]
      @summary = Summary.find(session[:id])
      @email_thread = @summary.email_thread
    else
      flash[:notice] = "You are logged out."
      redirect_to(:controller => '/account', :action => 'login')
      return
    end
    @emails = []
    email_count = 0
    for email in @email_thread.myemails
      email_count+=1
      s = email.Body
      #Keep track of tag beginning and ending
      open = false
      count = 1
      while s.include? "^"
        if !open
          case params[:control]
          when 'read' , 'sum' , 'show'
            s[s.index("^"),1] = sent(email_count, count)
          when 'label'
            s[s.index("^"),1] = label(email_count, count)
          end
          open = true
          count += 1
        else
          s[s.index("^"),1] = '</div></div>'
          open = false
        end
      end
      s.gsub! "/-\\", "^"
      @emails << email
    end
    #Render correct action
    case params[:control]
    when 'read'
      render :action => 'read'
    when 'sum'
      render :action => 'sum'
    when 'label'
      render :action => 'label'
    when 'show'
      render :action => 'show'
    end
  end
  
  #Buttons used to label speech acts, meta, and subjectivity
  def label(email_count, count)
"<div onmouseover='highlight(\"s#{email_count}.#{count}\");' onmouseout='lowlight(\"s#{email_count}.#{count}\");'>\
<img class='prop' id='prop#{email_count}.#{count}' onclick='toggleMe(\"prop#{email_count}.#{count}\");' src='/images/prop_d.gif' title='The sentence proposes a joint activity, or \"action item\".'>\
<img class='req' id='req#{email_count}.#{count}' onclick='toggleMe(\"req#{email_count}.#{count}\");' src='/images/req_d.gif' title='The sentence asks the recipient to perform an activity, or \"action item\".'>\
<img class='cmt' id='cmt#{email_count}.#{count}' onclick='toggleMe(\"cmt#{email_count}.#{count}\");' src='/images/cmt_d.gif' title='This sentence commits the user to some future course of action, or \"action item\".'>\
<img class='meet' id='meet#{email_count}.#{count}' onclick='toggleMe(\"meet#{email_count}.#{count}\");' src='/images/meet_d.gif' title='The sentence is regarding a joint activity in time or space.'>\
<img class='meta' id='meta#{email_count}.#{count}' onclick='toggleMe(\"meta#{email_count}.#{count}\");' src='/images/meta_d.gif' title='The sentence refers to this email thread.'>\
<img class='subj' id='subj#{email_count}.#{count}' onclick='toggleMe(\"subj#{email_count}.#{count}\");' src='/images/subj_d.gif' title='A sentence where the writer is expressing an opinion or strong sentiment.'>\
<div id='s#{email_count}.#{count}' class='clear_sentences'><b>#{email_count}.#{count}</b>&nbsp;"
  end
  
  #Div for wrapping sentences so they can be selected
  def sent(email_count, count)
    "<div><div id='s#{email_count}.#{count}' class='sentences'><b>#{email_count}.#{count}</b>&nbsp;"
  end
  
  #Update annotation progress differently depending on how far along the process is
  def update
    @summary = Summary.find(session[:id])
    #Give each study a timeout for security
    if Experiment.find(session[:exp]).still_valid?
      if !params[:summary][:Sent].blank?
        #Save summarization annotation
        if @summary.update_attributes(params[:summary])
          redirect_to :action => 'show', :control => 'label'
        else
          redirect_to :action => 'show', :control => 'sum'
        end
      else
        if !params[:summary][:Label].blank?
          #Save label annotation
          if @summary.update_attribute(:Label, params[:summary][:Label])
            redirect_to :action => 'selectSums'
          else
            redirect_to :action => 'show', :control => 'label'
          end
        else
          flash[:error] = 'You did not select any labels.'
          redirect_to :action => 'show', :control => 'label'
        end
      end
    else
      flash[:error] = 'This experiment cannot be changed anymore.'
      redirect_to :action => 'show', :control => 'sum'
    end
  end
  
  #Prompt to enter annotator's info
  def userinfo
    @participant = Participant.find(session[:user])
  end
  
  def create
    @participant = Participant.find(params[:id])
    if @participant.update_attributes(params[:participant])
      redirect_to :action => 'instructions'
    else
      render :action => 'userinfo'
    end
  end
  
  #Selects which thread to display for annotation
  def selectSums
    if (session[:first] == 'true')
      #Add Example as first thread
      session[:first] = nil
      @ex_thread = EmailThread.find(:first, :conditions => "Name = 'Example'")
      if @ex_thread.nil?
        flash[:error] ='Example thread does not exist'
        redirect_to :controller => '/account', :action => 'login'
        return
      end
      @summary = Summary.find(:first, :conditions => "email_thread_id = #{@ex_thread.id}")
      if @summary.nil?
        @summary = Summary.new
        @summary.email_thread = @ex_thread
        @summary.experiment = Experiment.find(:first)
        @summary.save
      end
      @summary.update_attribute('Sum', nil)
    else
      @summary = Summary.find_by_experiment_id(session[:exp], :order => 'list_order', :conditions => "Sent is null")
    end
    if @summary
      session[:id] = @summary.id
      
      redirect_to :action => 'show', :control => 'read'
    else
      reset_session
      flash[:notice] = "Thank you for participating in our study!"
      redirect_to :controller => '/account', :action => 'login'
    end
  end
  
  #Instructions window
  def instructions
    session[:first] = 'true'
  end
end
