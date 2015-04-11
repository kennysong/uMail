class ParticipantController < ApplicationController
  before_filter :login_required, :except => [:new, :create]
  layout 'framework'
  def index
    list
    render :action => 'list'
  end

  # GETs should be safe (see http://www.w3.org/2001/tag/doc/whenToUseGet.html)
  verify :method => :post, :only => [ :destroy, :create, :update ],
         :redirect_to => { :action => :list }

  def list
    @participants = Participant.find(:all)
  end

  def show
    @participant = Participant.find(params[:id])
  end

  def edit
    @participant = Participant.find(params[:id])
  end

  def update
    @participant = Participant.find(params[:id])
    if @participant.update_attributes(params[:participant])
      flash[:notice] = 'Participant was successfully updated.'
      redirect_to :action => 'show', :id => @participant
    else
      render :action => 'edit'
    end
  end

  def destroy
    Participant.find(params[:id]).destroy
    redirect_to :action => 'list'
  end
end
